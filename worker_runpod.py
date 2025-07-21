import os, json, requests, random, time, runpod
from urllib.parse import urlsplit

import torch
import torchaudio
import lightning as L
from torchvision.transforms import v2
from torio.io import StreamingMediaDecoder
from transformers import AutoProcessor
import torch.nn.functional as F
import tempfile
import subprocess
from moviepy.editor import VideoFileClip

from ThinkSound.models import create_model_from_config
from ThinkSound.models.utils import load_ckpt_state_dict
from ThinkSound.inference.sampling import sample, sample_discrete_euler
from data_utils.v2a_utils.feature_utils_224 import FeaturesUtils

def download_file(url, save_dir, file_name):
    os.makedirs(save_dir, exist_ok=True)
    file_suffix = os.path.splitext(urlsplit(url).path)[1]
    file_name_with_suffix = file_name + file_suffix
    file_path = os.path.join(save_dir, file_name_with_suffix)
    response = requests.get(url)
    response.raise_for_status()
    with open(file_path, 'wb') as file:
        file.write(response.content)
    return file_path

_CLIP_SIZE = 224
_CLIP_FPS = 8.0
_SYNC_SIZE = 224
_SYNC_FPS = 25.0

def pad_to_square(video_tensor):
    l, c, h, w = video_tensor.shape
    max_side = max(h, w)
    pad_h = max_side - h
    pad_w = max_side - w
    padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
    return F.pad(video_tensor, pad=padding, mode='constant', value=0)

class VGGSound:
    def __init__(self, sample_rate: int = 44_100, duration_sec: float = 9.0):
        self.sample_rate = sample_rate
        self.duration_sec = duration_sec
        self.audio_samples = int(sample_rate * duration_sec)
        self.clip_expected_length = int(_CLIP_FPS * duration_sec)
        self.sync_expected_length = int(_SYNC_FPS * duration_sec)
        self.clip_transform = v2.Compose([
            v2.Lambda(pad_to_square),
            v2.Resize((_CLIP_SIZE, _CLIP_SIZE), interpolation=v2.InterpolationMode.BICUBIC),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])
        self.clip_processor = AutoProcessor.from_pretrained("facebook/metaclip-h14-fullcc2.5b")
        self.sync_transform = v2.Compose([
            v2.Resize(_SYNC_SIZE, interpolation=v2.InterpolationMode.BICUBIC),
            v2.CenterCrop(_SYNC_SIZE),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def sample(self, video_path, label, cot):
        reader = StreamingMediaDecoder(video_path)
        reader.add_basic_video_stream(frames_per_chunk=self.clip_expected_length, frame_rate=_CLIP_FPS, format='rgb24')
        reader.add_basic_video_stream(frames_per_chunk=self.sync_expected_length, frame_rate=_SYNC_FPS, format='rgb24')
        reader.fill_buffer()
        clip_chunk, sync_chunk = reader.pop_chunks()
        if sync_chunk is None:
            raise RuntimeError(f'Sync video returned None {video_path}')
        
        clip_chunk = clip_chunk[:self.clip_expected_length]
        if clip_chunk.shape[0] != self.clip_expected_length:
            padding_needed = self.clip_expected_length - clip_chunk.shape[0]
            if padding_needed > 0:
                clip_chunk = torch.cat((clip_chunk, clip_chunk[-1].repeat(padding_needed, 1, 1, 1)), dim=0)
        
        clip_chunk = self.clip_processor(images=pad_to_square(clip_chunk), return_tensors="pt")["pixel_values"]
        sync_chunk = sync_chunk[:self.sync_expected_length]
        if sync_chunk.shape[0] != self.sync_expected_length:
            sync_chunk = torch.cat((sync_chunk, sync_chunk[-1].repeat(self.sync_expected_length - sync_chunk.shape[0], 1, 1, 1)), dim=0)
        sync_chunk = self.sync_transform(sync_chunk)
        
        return {
            'id': video_path,
            'caption': label,
            'caption_cot': cot,
            'clip_video': clip_chunk,
            'sync_video': sync_chunk,
        }

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
extra_device = 'cuda:1' if torch.cuda.device_count() > 1 else device
print(f"Loading on device {device}")

# Define local paths to model checkpoints
vae_ckpt = "ckpts/vae.ckpt"
synchformer_ckpt = "ckpts/synchformer_state_dict.pth"
thinksound_ckpt = "ckpts/thinksound_light.ckpt"

# Initialize feature extractor
feature_extractor = FeaturesUtils(
    vae_ckpt=None,
    vae_config='ThinkSound/configs/model_configs/stable_audio_2_0_vae.json',
    enable_conditions=True,
    synchformer_ckpt=synchformer_ckpt
).eval().to(extra_device)

# Load diffusion model
with open("ThinkSound/configs/model_configs/thinksound.json") as f:
    model_config = json.load(f)
diffusion_model = create_model_from_config(model_config)
diffusion_model.load_state_dict(torch.load(thinksound_ckpt))
diffusion_model.to(device)
diffusion_model.pretransform.load_state_dict(load_ckpt_state_dict(vae_ckpt, prefix='autoencoder.'))

def get_video_duration(video_path):
    return VideoFileClip(video_path).duration

def set_manual_seed(seed):    
    # PyTorch (CPU and GPU)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # For CUDA devices
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    
    # Ensure deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@torch.inference_mode()
def generate(input):
    try:
        values = input["input"]

        input_video = values['input_video'] 
        input_video = download_file(url=input_video, save_dir='/content', file_name='input_video')
        caption = ""
        cot = None
        output_path = values['output_path']
        seed = values['seed']
        if seed == 0:
            random.seed(int(time.time()))
            seed = random.randint(0, 18446744073709551615)
        set_manual_seed(seed)

        cot = cot or caption
        duration_sec = get_video_duration(input_video)
        preprocesser = VGGSound(duration_sec=duration_sec)
        data = preprocesser.sample(input_video, caption, cot)

        preprocessed_data = {
            'metaclip_global_text_features': feature_extractor.encode_text(data['caption'])[0].detach().cpu().squeeze(0),
            'metaclip_text_features': feature_extractor.encode_text(data['caption'])[1].detach().cpu().squeeze(0),
            't5_features': feature_extractor.encode_t5_text(data['caption_cot']).detach().cpu().squeeze(0),
            'metaclip_features': feature_extractor.encode_video_with_clip(data['clip_video'].unsqueeze(0).to(extra_device)).detach().cpu().squeeze(0),
            'sync_features': feature_extractor.encode_video_with_sync(data['sync_video'].unsqueeze(0).to(extra_device)).detach().cpu().squeeze(0),
            'video_exist': torch.tensor(True)
        }

        sync_seq_len = preprocessed_data['sync_features'].shape[0]
        clip_seq_len = preprocessed_data['metaclip_features'].shape[0]
        latent_seq_len = int(194/9 * duration_sec)
        diffusion_model.model.model.update_seq_lengths(latent_seq_len, clip_seq_len, sync_seq_len)

        metadata = [preprocessed_data]
        with torch.amp.autocast(device):
            conditioning = diffusion_model.conditioner(metadata, device)
            conditioning['metaclip_features'][~torch.stack([item['video_exist'] for item in metadata])] = diffusion_model.model.model.empty_clip_feat
            conditioning['sync_features'][~torch.stack([item['video_exist'] for item in metadata])] = diffusion_model.model.model.empty_sync_feat
            cond_inputs = diffusion_model.get_conditioning_inputs(conditioning)
            noise = torch.randn([1, diffusion_model.io_channels, latent_seq_len]).to(device)
            fakes = sample_discrete_euler(diffusion_model.model, noise, 24, **cond_inputs, cfg_scale=5, batch_cfg=True)
            if diffusion_model.pretransform is not None:
                fakes = diffusion_model.pretransform.decode(fakes)

        audios = fakes.to(torch.float32).div(torch.max(torch.abs(fakes))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_audio:
            torchaudio.save(tmp_audio.name, audios[0], 44100)
            audio_path = tmp_audio.name

        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_video:
            output_video_path = output_path if output_path else tmp_video.name
        subprocess.run([
            'ffmpeg', '-y', '-i', input_video, '-i', audio_path,
            '-c:v', 'copy', '-map', '0:v:0', '-map', '1:a:0',
            '-shortest', output_video_path
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        os.unlink(audio_path)

        result = output_video_path

        notify_uri = values['notify_uri']
        del values['notify_uri']
        notify_token = values['notify_token']
        del values['notify_token']
        discord_id = values['discord_id']
        del values['discord_id']
        if(discord_id == "discord_id"):
            discord_id = os.getenv('com_camenduru_discord_id')
        discord_channel = values['discord_channel']
        del values['discord_channel']
        if(discord_channel == "discord_channel"):
            discord_channel = os.getenv('com_camenduru_discord_channel')
        discord_token = values['discord_token']
        del values['discord_token']
        if(discord_token == "discord_token"):
            discord_token = os.getenv('com_camenduru_discord_token')
        job_id = values['job_id']
        del values['job_id']
        with open(result, 'rb') as file:
            response = requests.post("https://upload.tost.ai/api/v1", files={'file': file})
        response.raise_for_status()
        result_url = response.text
        notify_payload = {"jobId": job_id, "result": result_url, "status": "DONE"}
        web_notify_uri = os.getenv('com_camenduru_web_notify_uri')
        web_notify_token = os.getenv('com_camenduru_web_notify_token')
        if(notify_uri == "notify_uri"):
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
        else:
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            requests.post(notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        return {"jobId": job_id, "result": result_url, "status": "DONE"}
    except Exception as e:
        error_payload = {"jobId": job_id, "status": "FAILED"}
        try:
            if(notify_uri == "notify_uri"):
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            else:
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
                requests.post(notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        except:
            pass
        return {"jobId": job_id, "result": f"FAILED: {str(e)}", "status": "FAILED"}
    finally:
        if os.path.exists(result):
            os.remove(result)

runpod.serverless.start({"handler": generate})