FROM anyscale/ray-llm:2.55.1-py311-cu128

RUN pip install --no-cache-dir --upgrade "torch==2.10.0" "torchvision==0.25.0" "matplotlib==3.10.1" "diffusers==0.32.2" "transformers==4.57.4" "accelerate==1.5.2" "xgboost==2.1.4" "pytorch-lightning==2.5.1" "pyarrow==19.0.1" "datasets==3.5.0" "evaluate==0.4.3" "scikit-learn==1.6.1" "torch-tb-profiler==0.4.3" "tensorboard==2.19.0" "sentence-transformers==5.2.2" "textdistance==4.6.3" "vllm[runai]==0.18.0" "pandas==3.0.3"

RUN sudo apt-get update -y \
    && sudo apt-get install --no-install-recommends -y wget \
    && sudo rm -f /etc/apt/sources.list.d/*
