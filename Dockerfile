FROM ubuntu:22.04

WORKDIR /content

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=True
ENV PYTHONDONTWRITEBYTECODE=True
ENV PATH="/home/camenduru/.local/bin:/usr/local/cuda/bin:${PATH}"

RUN apt update -y && apt install -y software-properties-common build-essential \
    libgl1 libglib2.0-0 zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev && \
    add-apt-repository -y ppa:git-core/ppa && apt update -y && \
    apt install -y python-is-python3 python3-pip sudo nano aria2 curl wget git git-lfs unzip unrar ffmpeg pkg-config && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://developer.download.nvidia.com/compute/cuda/12.9.1/local_installers/cuda_12.9.1_575.57.08_linux.run -d /content -o cuda_12.9.1_575.57.08_linux.run && sh cuda_12.9.1_575.57.08_linux.run --silent --toolkit && \
    echo "/usr/local/cuda/lib64" >> /etc/ld.so.conf && ldconfig && \
    git clone https://github.com/aristocratos/btop /content/btop && cd /content/btop && make && make install && \
    adduser --disabled-password --gecos '' camenduru && \
    adduser camenduru sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    chown -R camenduru:camenduru /content && \
    chmod -R 777 /content && \
    chown -R camenduru:camenduru /home && \
    chmod -R 777 /home
    
USER camenduru

RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu128 && \
    pip install xformers --extra-index-url https://download.pytorch.org/whl/cu128 && \
    pip install https://github.com/camenduru/wheels/releases/download/3090/flash_attn-2.8.1-cp310-cp310-linux_x86_64.whl && \
    pip install lightning transformers==4.53.2 moviepy==1.0.3 k-diffusion open-clip-torch omegaconf blobfile tiktoken sentencepiece descript-audio-codec vector-quantize-pytorch protobuf==5.29.5 runpod && \
    pip install git+https://github.com/patrick-kidger/torchcubicspline && \
    pip install git+https://github.com/junjun3518/alias-free-torch && \
    GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --branch docker https://github.com/camenduru/ThinkSound-hf /content/ThinkSound && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/ThinkSound/resolve/main/thinksound_light.ckpt -d /content/ThinkSound/ckpts -o thinksound_light.ckpt && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/ThinkSound/resolve/main/vae.ckpt -d /content/ThinkSound/ckpts -o vae.ckpt && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/ThinkSound/resolve/main/synchformer_state_dict.pth -d /content/ThinkSound/ckpts -o synchformer_state_dict.pth && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/facebook/metaclip-h14-fullcc2.5b/raw/main/special_tokens_map.json -d /content/ThinkSound/ckpts/metaclip-h14-fullcc2.5b -o special_tokens_map.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/facebook/metaclip-h14-fullcc2.5b/raw/main/added_tokens.json -d /content/ThinkSound/ckpts/metaclip-h14-fullcc2.5b -o added_tokens.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/facebook/metaclip-h14-fullcc2.5b/raw/main/merges.txt -d /content/ThinkSound/ckpts/metaclip-h14-fullcc2.5b -o merges.txt && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/facebook/metaclip-h14-fullcc2.5b/raw/main/vocab.json -d /content/ThinkSound/ckpts/metaclip-h14-fullcc2.5b -o vocab.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/facebook/metaclip-h14-fullcc2.5b/raw/main/tokenizer_config.json -d /content/ThinkSound/ckpts/metaclip-h14-fullcc2.5b -o tokenizer_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/facebook/metaclip-h14-fullcc2.5b/raw/main/preprocessor_config.json -d /content/ThinkSound/ckpts/metaclip-h14-fullcc2.5b -o preprocessor_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/facebook/metaclip-h14-fullcc2.5b/resolve/main/model.safetensors -d /content/ThinkSound/ckpts/metaclip-h14-fullcc2.5b -o model.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/facebook/metaclip-h14-fullcc2.5b/raw/main/config.json -d /content/ThinkSound/ckpts/metaclip-h14-fullcc2.5b -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/google/t5-v1_1-xl/raw/main/config.json -d /content/ThinkSound/ckpts/t5-v1_1-xl -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/google/t5-v1_1-xl/raw/main/generation_config.json -d /content/ThinkSound/ckpts/t5-v1_1-xl -o generation_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/google/t5-v1_1-xl/raw/main/special_tokens_map.json -d /content/ThinkSound/ckpts/t5-v1_1-xl -o special_tokens_map.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/google/t5-v1_1-xl/resolve/main/spiece.model -d /content/ThinkSound/ckpts/t5-v1_1-xl -o spiece.model && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/google/t5-v1_1-xl/raw/main/tokenizer_config.json -d /content/ThinkSound/ckpts/t5-v1_1-xl -o tokenizer_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/google/t5-v1_1-xl/resolve/main/pytorch_model.bin -d /content/ThinkSound/ckpts/t5-v1_1-xl -o pytorch_model.bin

COPY ./worker_runpod.py /content/ThinkSound/worker_runpod.py
WORKDIR /content/ThinkSound
CMD python worker_runpod.py