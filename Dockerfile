FROM nvidia/cuda:11.7.1-runtime-ubuntu22.04

# To use a different model, change the model URL below:
ARG MODEL_URL='https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt'

# If you are using a private Huggingface model (sign in required to download) insert your Huggingface
# access token (https://huggingface.co/settings/tokens) below:
ARG HF_TOKEN='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'

RUN apt update && apt-get -y install git wget \
    python3.10 python3.10-venv python3-pip \
    build-essential libgl-dev libglib2.0-0 vim \
    git-lfs
RUN ln -s /usr/bin/python3.10 /usr/bin/python

RUN useradd -ms /bin/bash banana
WORKDIR /app

RUN git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git && \
    cd stable-diffusion-webui && \
    git checkout 3e0f9a75438fa815429b5530261bcf7d80f3f101
WORKDIR /app/stable-diffusion-webui

ENV MODEL_URL=${MODEL_URL}
ENV HF_TOKEN=${HF_TOKEN}

RUN pip install tqdm requests
ADD download_checkpoint.py .
RUN python download_checkpoint.py

ADD prepare.py .
RUN python prepare.py --skip-torch-cuda-test --xformers --reinstall-torch --reinstall-xformers

ADD download.py download.py
RUN python download.py --use-cpu=all

RUN pip install dill

RUN mkdir -p extensions/banana/scripts
ADD script.py extensions/banana/scripts/banana.py
RUN git lfs install
RUN git clone https://github.com/Mikubill/sd-webui-controlnet extensions/sd-webui-controlnet
RUN git clone https://huggingface.co/kohya-ss/ControlNet-diff-modules
RUN mv ControlNet-diff-modules/*.safetensors extensions/sd-webui-controlnet/models

#Download Lora
RUN wget -P models/Lora/add_detail.safetensors https://civitai.com/api/download/models/62833
RUN wget -P models/Lora/ArmorSuit_v1.safetensors https://civitai.com/api/download/models/63688
RUN wget -P models/Lora/arcane.safetensors https://civitai.com/api/download/models/8339 
RUN wget -P models/Lora/attire_superman.safetensors https://civitai.com/api/download/models/88747 
RUN wget -P models/Lora/EnergyVeins.safetensors https://civitai.com/api/download/models/89147 
RUN wget -P models/Lora/Gigachad.safetensors https://civitai.com/api/download/models/21518
RUN wget -P models/Lora/insanobot.safetensors https://civitai.com/api/download/models/95971
RUN wget -P models/Lora/mechaarmor.safetensors https://civitai.com/api/download/models/99186
RUN wget -P models/Lora/more_details.safetensors https://civitai.com/api/download/models/87153 
RUN wget -P models/Lora/sam_yang.safetensors https://civitai.com/api/download/models/7804 
RUN wget -P models/Lora/Sy3.safetensors https://civitai.com/api/download/models/101470 
RUN wget -P models/Lora/cute_cat.safetensors https://civitai.com/api/download/models/106565 
RUN wget -P models/Lora/cute_dog.safetensors https://civitai.com/api/download/models/106575 
RUN wget -P models/Lora/dc_marvel.safetensors https://civitai.com/api/download/models/10580 
RUN wget -P models/Lora/epi_noiseoffset2.safetensors https://civitai.com/api/download/models/16576
RUN wget -P models/Lora/ink_scenery.safetensors https://civitai.com/api/download/models/83390
RUN wget -P models/Lora/OilPaintStyle.safetensors https://civitai.com/api/download/models/47588 
RUN wget -P models/Lora/reelmech1v2.safetensors https://civitai.com/api/download/models/85371
RUN wget -P models/Lora/retrowave.safetensors https://civitai.com/api/download/models/77964
RUN wget -P models/Lora/Smoke.safetensors https://civitai.com/api/download/models/67323
RUN wget -P models/Lora/Unholy.safetensors https://civitai.com/api/download/models/81311
RUN wget -P models/Lora/sadcatmeme.safetensors https://civitai.com/api/download/models/33501
RUN wget -P models/Lora/therockmeme.safetensors https://civitai.com/api/download/models/104539
RUN wget -P models/Lora/surprise_meme.safetensors https://civitai.com/api/download/models/113216
RUN wget -P models/Lora/SaltBaeMeme.safetensors https://civitai.com/api/download/models/114063

#Download Upscale
RUN wget -P models/ESRGAN/ https://huggingface.co/lokCX/4x-Ultrasharp/resolve/main/4x-UltraSharp.pth

#Add Negative
RUN wget -P embeddings/BadDream.pt https://civitai.com/api/download/models/77169
RUN wget -P embeddings/bad_artist.pt https://civitai.com/api/download/models/6056
RUN wget -P embeddings/badhandv4.pt https://civitai.com/api/download/models/20068
RUN wget -P embeddings/negative_hand-neg.pt https://civitai.com/api/download/models/60938


ADD app.py app.py
ADD server.py server.py

CMD ["python", "server.py", "--xformers", "--disable-safe-unpickle", "--lowram", "--no-hashing", "--listen", "--port", "8000"]