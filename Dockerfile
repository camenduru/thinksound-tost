FROM ubuntu:22.04

WORKDIR /content

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=True
ENV PYTHONDONTWRITEBYTECODE=True
ENV PATH="/home/camenduru/.local/bin:/usr/local/cuda/bin:${PATH}"

RUN apt update -y && apt install -y software-properties-common build-essential \
    libgl1 libglib2.0-0 zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev && \
    add-apt-repository -y ppa:git-core/ppa && apt update -y && \
    apt install -y python-is-python3 python3-pip sudo nano aria2 curl wget git git-lfs unzip unrar ffmpeg && \
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
    pip install flash-attn --no-build-isolation && \
    pip install lightning transformers==4.53.2 moviepy==1.0.3 k-diffusion open-clip-torch omegaconf blobfile tiktoken sentencepiece descript-audio-codec vector-quantize-pytorch && \
    pip install git+https://github.com/patrick-kidger/torchcubicspline && \
    pip install git+https://github.com/junjun3518/alias-free-torch && \
    git clone --depth 1 --branch dev https://github.com/camenduru/ThinkSound-hf /content/ThinkSound && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/ThinkSound/resolve/main/thinksound_light.ckpt -d /content/ThinkSound/ckpts -o thinksound_light.ckpt && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/ThinkSound/resolve/main/vae.ckpt -d /content/ThinkSound/ckpts -o vae.ckpt && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/ThinkSound/resolve/main/synchformer_state_dict.pth -d /content/ThinkSound/ckpts -o synchformer_state_dict.pth

COPY ./worker_runpod.py /content/ComfyUI/worker_runpod.py
WORKDIR /content/ThinkSound
CMD python worker_runpod.py