FROM anyscale/ray:2.53.0-slim-py311-cu128
# =============================================================================
# Isaac Sim 5.1.0 + Isaac Lab on Anyscale Ray
# =============================================================================
# Key constraints:
#   - Isaac Sim 5.X requires Python 3.11 (NOT 3.10, NOT 3.12)
#   - Isaac Lab recommends torch 2.7.0 + CUDA 12.8
#   - Headless GPU rendering needs Vulkan + EGL userspace libs
# =============================================================================
ENV DEBIAN_FRONTEND=noninteractive
ENV OMNI_KIT_ACCEPT_EULA=YES
ENV ACCEPT_EULA=Y
ENV PYTHONUNBUFFERED=1
# ---------- headless rendering ----------
ENV DISPLAY=""
ENV OMNI_KIT_RENDERING_MODE=headless
# Force EGL over GLX (no X server on Anyscale nodes)
ENV __EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d
# Vulkan ICD: NVIDIA driver mounts this at runtime
ENV VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
ENV VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json
# Request all driver capabilities (also injects the graphics device nodes: nvidia-modeset,
# /dev/dri/renderD*). NOTE: on this platform this is NOT sufficient by itself; the host
# driver carries no graphics *userspace*, so the NVIDIA container runtime has no graphics libs
# to mount regardless of this setting. The graphics userspace is baked in explicitly below.
ENV NVIDIA_DRIVER_CAPABILITIES=all
USER root
# ---------- system deps: EGL + Vulkan + misc ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    ca-certificates \
    # EGL / OpenGL
    libgl1 \
    libegl1 \
    libegl-mesa0 \
    libgles2 \
    libglvnd0 \
    libglvnd-dev \
    # Vulkan (required for Isaac Sim GPU rendering)
    libvulkan1 \
    libvulkan-dev \
    vulkan-tools \
    mesa-vulkan-drivers \
    # X11 libs (Isaac Sim links against these even headless)
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    libxi6 \
    libxkbcommon0 \
    libx11-6 \
    libxext6 \
    libxt6 \
    libglu1-mesa \
    # Misc
    libsm6 \
    libice6 \
    libfontconfig1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
# Create NVIDIA Vulkan ICD file (in case NVIDIA container runtime doesn't mount it)
RUN mkdir -p /etc/vulkan/icd.d && \
    echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3.277"}}' \
    > /etc/vulkan/icd.d/nvidia_icd.json
# ---------- NVIDIA graphics userspace (added 2026-07-22) ----------
# The NVIDIA container runtime injects only COMPUTE driver libs (libcuda, nvidia-ml, ...); the
# host driver carries no graphics userspace, so libGLX_nvidia.so.0 (the NVIDIA Vulkan ICD),
# libnvidia-glcore/rtcore/eglcore/... and the GLVND EGL vendor JSON are all absent from the
# container. Without them Isaac Sim's Vulkan/RTX renderer can't initialize
# (vkCreateInstance -> ERROR_INCOMPATIBLE_DRIVER) -> all-black frames + PhysX-GPU init hang.
# Fix: bake the *graphics* userspace from the version-matched (=host kernel driver) .run.
# The EGL vendor JSON (10_nvidia.json) is essential; the driver's init path goes through GLVND
# EGL, and without it registered the init silently fails (VK_ERROR_INITIALIZATION_FAILED).
# Verified live on g7e.4xlarge / RTX PRO 6000 Blackwell: vulkaninfo then enumerates the GPU.
ENV NV_DRIVER_VERSION=580.126.09
RUN set -eux; cd /tmp; \
    curl -fsSL -o nv.run \
      "https://us.download.nvidia.com/tesla/${NV_DRIVER_VERSION}/NVIDIA-Linux-x86_64-${NV_DRIVER_VERSION}.run"; \
    sh nv.run --extract-only --target /tmp/nvx; \
    L=/usr/lib/x86_64-linux-gnu; \
    cp -a \
      "/tmp/nvx/libGLX_nvidia.so.${NV_DRIVER_VERSION}" \
      "/tmp/nvx/libEGL_nvidia.so.${NV_DRIVER_VERSION}" \
      "/tmp/nvx/libnvidia-glcore.so.${NV_DRIVER_VERSION}" \
      "/tmp/nvx/libnvidia-glsi.so.${NV_DRIVER_VERSION}" \
      "/tmp/nvx/libnvidia-glvkspirv.so.${NV_DRIVER_VERSION}" \
      "/tmp/nvx/libnvidia-gpucomp.so.${NV_DRIVER_VERSION}" \
      "/tmp/nvx/libnvidia-rtcore.so.${NV_DRIVER_VERSION}" \
      "/tmp/nvx/libnvidia-eglcore.so.${NV_DRIVER_VERSION}" \
      "/tmp/nvx/libnvidia-tls.so.${NV_DRIVER_VERSION}" \
      "/tmp/nvx/libnvidia-allocator.so.${NV_DRIVER_VERSION}" \
      /tmp/nvx/libnvidia-egl-gbm.so.1.* \
      "$L/"; \
    ln -sf "libGLX_nvidia.so.${NV_DRIVER_VERSION}" "$L/libGLX_nvidia.so.0"; \
    ln -sf "libEGL_nvidia.so.${NV_DRIVER_VERSION}" "$L/libEGL_nvidia.so.0"; \
    ln -sf "libnvidia-allocator.so.${NV_DRIVER_VERSION}" "$L/libnvidia-allocator.so.1"; \
    ln -sf "$(cd "$L" && ls libnvidia-egl-gbm.so.1.*)" "$L/libnvidia-egl-gbm.so.1"; \
    mkdir -p /usr/share/glvnd/egl_vendor.d /usr/share/egl/egl_external_platform.d; \
    cp /tmp/nvx/10_nvidia.json /usr/share/glvnd/egl_vendor.d/; \
    cp /tmp/nvx/15_nvidia_gbm.json /tmp/nvx/20_nvidia_xcb.json \
       /tmp/nvx/20_nvidia_xlib.json /tmp/nvx/10_nvidia_wayland.json \
       /usr/share/egl/egl_external_platform.d/; \
    ldconfig; \
    rm -rf /tmp/nv.run /tmp/nvx
USER ray
WORKDIR /home/ray
# ---------- PyTorch (pinned to Isaac Lab's recommended version) ----------
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir \
    "torch==2.7.0" \
    "torchvision==0.22.0" \
    --index-url https://download.pytorch.org/whl/cu128
# ---------- Isaac Sim 5.1.0 (skip extscache to save ~10GB, lazy-loads at runtime) ----------
RUN python -m pip install --no-cache-dir \
    "isaacsim[all]==5.1.0" \
    --extra-index-url https://pypi.nvidia.com
# ---------- Isaac Lab from source (pinned to an exact commit) ----------
# main@b0542fe == extension version 0.54.4, the tree this image was verified against.
# Pinned by SHA instead of tracking `main`, which moves. Single-commit shallow fetch
# (GitHub serves a SHA directly), so the checkout stays ~34 MB.
#
# Do NOT "pin" this to the v2.3.2 tag: that tag requires flatdict==4.0.1, which is
# sdist-only on PyPI, and its setup.py does `import pkg_resources` -- removed from modern
# setuptools -- so the editable install below dies with ModuleNotFoundError. `main` requires
# flatdict>=4.1.0, which ships a wheel and needs no build step.
# Do NOT move to `develop` / v3.0.0-beta* either: that line requires Python >=3.12, and
# Isaac Sim 5.1.0 needs 3.11.
ENV ISAACLAB_COMMIT=b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8
RUN mkdir -p /home/ray/IsaacLab && cd /home/ray/IsaacLab && \
    git init -q . && \
    git remote add origin https://github.com/isaac-sim/IsaacLab.git && \
    git fetch -q --depth 1 origin "${ISAACLAB_COMMIT}" && \
    git checkout -q FETCH_HEAD
# Install Isaac Lab extensions as editable packages
RUN cd /home/ray/IsaacLab && \
    python -m pip install --no-cache-dir \
    -e source/isaaclab \
    -e source/isaaclab_tasks \
    -e source/isaaclab_rl
# ---------- Training deps ----------
RUN python -m pip install --no-cache-dir \
    pillow \
    wandb \
    gymnasium
# ---------- PYTHONPATH fallback (belt + suspenders with editable installs above) ----------
ENV PYTHONPATH=/home/ray/IsaacLab/source/isaaclab:\
/home/ray/IsaacLab/source/isaaclab_tasks:\
/home/ray/IsaacLab/source/isaaclab_rl:\
${PYTHONPATH}
WORKDIR /home/ray/default

# =============================================================================
# VLA / lerobot overlay (added 2026-05-05 -- verified compatible with Isaac Sim)
# =============================================================================

USER root
# torchcodec needs libav* shared libs at runtime; lerobot decodes mp4 frames
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*
USER ray

# lerobot transitively requires rerun-sdk which requires numpy>=2 -- incompatible
# with Isaac Sim's compiled ABI. Install lerobot --no-deps and provide its
# runtime deps explicitly with numpy<2 preserved.
RUN python -m pip install --no-cache-dir --no-deps lerobot==0.4.3

RUN python -m pip install --no-cache-dir \
    "numpy>=1.26,<2" \
    "datasets>=4.0,<4.2" \
    "accelerate>=1.10,<2" \
    "deepdiff>=7,<9" \
    "diffusers>=0.27.2,<0.36" \
    "draccus==0.10.0" \
    "einops>=0.8,<0.9" \
    "huggingface-hub[cli,hf-transfer]>=0.34.2,<0.36" \
    "imageio[ffmpeg]==2.37.0" \
    "jsonlines>=4,<5" \
    "packaging>=24.2,<26" \
    "pyserial>=3.5,<4" \
    "sentencepiece==0.2.2" \
    "termcolor>=2.4,<4" \
    "torchcodec==0.5" \
    "av==15.1.0" \
    "num2words>=0.5,<0.6" \
    "wandb>=0.24,<0.25" \
    "s3fs>=2024.1" \
    "fsspec[s3]>=2024.1"

# Patched transformers fork (huggingface/transformers@fix/lerobot_openpi).
# PI0.5's `lerobot/pi05_base` checkpoint stores Gemma layernorm parameters
# under a different key layout than mainline 4.57+, AND PI05Pytorch.__init__
# aborts unless `transformers.models.siglip.check` exists (only in this fork).
# Install with deps so tokenizers gets pulled to the fork's required <0.22.
# Pinned to the branch tip as of 2026-07-27 (reports itself as transformers 4.53.3);
# a bare branch name would let a rebase or force-push silently change the model code.
RUN python -m pip uninstall -y transformers tokenizers || true \
 && python -m pip install --no-cache-dir \
    "git+https://github.com/huggingface/transformers.git@dcddb970176382c0fcf4521b0c0e6fc15894dfe0"

# =============================================================================
# Bake the PaliGemma tokenizer into the image HF cache (added 2026-07-20).
# PI0.5's preprocessor calls AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224").
# We stage the ~22 MB of tokenizer + config files (NO model weights) from the tutorial's
# PUBLIC S3 mirror using s3fs in ANONYMOUS mode -- so this build needs **NO Hugging Face
# token and NO credentials**. Every attendee can build this image with zero secrets.
# (The presenter uploaded those files once; the same S3 prefix carries GEMMA_NOTICE.txt
#  with the Gemma Terms of Use under which they are redistributed.) At runtime,
#  HF_HUB_OFFLINE=1 loads the tokenizer from this baked cache -- the tutorial touches
#  Hugging Face nowhere, at build time or run time.
# (All other artifacts -- datasets + the PI0.5 model -- stream from the same public S3
#  mirror at runtime, so this tokenizer is the only thing baked into the image.)
# =============================================================================
RUN python -c "import os, s3fs; \
dst=os.path.expanduser('~/.cache/huggingface/hub')+'/'; os.makedirs(dst, exist_ok=True); \
s3fs.S3FileSystem(anon=True).get('anyscale-public-materials-use2/ray_summit_robotics_2026/paligemma_tokenizer/hub/', dst, recursive=True)" \
 && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -c "from transformers import AutoTokenizer, AutoProcessor; \
AutoTokenizer.from_pretrained('google/paligemma-3b-pt-224'); \
AutoProcessor.from_pretrained('google/paligemma-3b-pt-224'); \
print('PaliGemma tokenizer+processor cached OK (offline, tokenless)')"

# Sanity check (fails the build if the env is broken)
RUN python -c "import lerobot, transformers, isaaclab; \
from lerobot.policies.pi05.modeling_pi05 import PI05Policy, PI05Config; \
PI05Config(); \
print(f'lerobot={lerobot.__version__} transformers={transformers.__version__} OK')"