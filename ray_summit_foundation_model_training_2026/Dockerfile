FROM anyscale/ray:2.55.1-py311-cu128
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

RUN pip install --upgrade pip && \
    pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
      torch==2.9.1 \
      torchvision==0.24.1 \
      transformers==4.48.0 \
      datasets==2.21.0 \
      deepspeed==0.18.9 \
      accelerate==1.3.0 \
      torchmetrics==1.6.1 \
      hf_transfer==0.1.8