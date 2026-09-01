"""Shared helpers for the *Foundation Model Distributed Training with Ray* course.

Kept deliberately small. The teaching-critical Ray calls (`prepare_model`,
`get_dataset_shard`, `ray.train.report`, checkpoint save/load, `fully_shard`,
`deepspeed.initialize`, ...) live *inline* in the notebooks so they're visible.
This module only holds boilerplate every notebook repeats: mixed-precision
selection, the standard Ray `runtime_env`, a couple of model builders, and a
cluster pretty-printer.

All notebooks ship this directory to workers via `runtime_env={"working_dir": "."}`,
so `import common.utils` resolves on every worker.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional

import torch


# ---------------------------------------------------------------------------
# Mixed precision
# ---------------------------------------------------------------------------
# Lesson baked in here: on an NVIDIA T4 (Turing, sm_75), `torch.cuda.
# is_bf16_supported()` returns **True** because recent PyTorch counts *software
# emulation*. But T4 has no *native* bf16 tensor cores — emulated bf16 is slow,
# while fp16 runs on real tensor cores. So we pick precision by native support
# (compute capability >= 8.0, i.e. Ampere+), NOT by the naive helper.


def has_native_bf16() -> bool:
    """True only if the current CUDA device has *native* bf16 (Ampere+ / sm_80+).

    On a T4 this returns False even though `torch.cuda.is_bf16_supported()`
    returns True (the latter includes slow software emulation).
    """
    if not torch.cuda.is_available():
        return False
    try:
        # Newer PyTorch exposes the native-only check directly.
        return torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:
        # Older PyTorch: fall back to a compute-capability gate.
        major, _ = torch.cuda.get_device_capability()
        return major >= 8


def mixed_precision_dtype() -> torch.dtype:
    """Return the dtype to train in: bf16 on Ampere+, fp16 on T4-class GPUs."""
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if has_native_bf16() else torch.float16


def deepspeed_precision_config() -> dict[str, Any]:
    """DeepSpeed config fragment for mixed precision, matched to the GPU.

    bf16 on Ampere+; fp16 (with DeepSpeed's loss scaler) on T4-class GPUs.
    """
    if has_native_bf16():
        return {"bf16": {"enabled": True}, "grad_accum_dtype": "bf16"}
    # T4: fp16. DeepSpeed manages dynamic loss scaling automatically.
    return {"fp16": {"enabled": True}}


# ---------------------------------------------------------------------------
# Ray runtime environment
# ---------------------------------------------------------------------------
# Dependencies (torch, transformers, deepspeed, ...) are installed cluster-wide
# via the Anyscale workspace, so we do NOT pip-install in the runtime_env. We
# only (1) ship this working directory so `common/` imports resolve on workers,
# and (2) set the env vars every worker needs.

# NCCL flags for g4dn: these T4s talk over PCIe (no NVLink) and the instances
# use AWS ENA networking (no InfiniBand). Disabling P2P and IB steers NCCL onto
# transports that initialize cleanly on this containerized, PCIe-only cluster.
DEFAULT_ENV_VARS: dict[str, str] = {
    "RAY_TRAIN_V2_ENABLED": "1",
    "NCCL_P2P_DISABLE": "1",
    "NCCL_IB_DISABLE": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    # 0% Hugging Face at runtime. Every model and dataset is staged from the
    # tutorial's public S3 mirror, so these force offline mode on every worker:
    # an accidental repo-id load raises instead of silently hitting the Hub
    # (which would throttle under a room full of identical clusters).
    "HF_HUB_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
}


def build_runtime_env(
    extra_env: Optional[dict[str, str]] = None,
    working_dir: str = ".",
) -> dict[str, Any]:
    """Build the standard `runtime_env` for this course.

    Ships `working_dir` (so `import common.utils` works on every worker) and the
    default env vars. Pass `extra_env` to add or override (e.g. an `HF_TOKEN`).
    No `pip` key: deps are already installed on every node by the workspace.
    """
    env_vars = dict(DEFAULT_ENV_VARS)
    if extra_env:
        env_vars.update(extra_env)
    # Pass through HF_TOKEN from the driver if present (gated models).
    if "HF_TOKEN" not in env_vars and os.environ.get("HF_TOKEN"):
        env_vars["HF_TOKEN"] = os.environ["HF_TOKEN"]
    return {"working_dir": working_dir, "env_vars": env_vars}


# ---------------------------------------------------------------------------
# Cluster introspection
# ---------------------------------------------------------------------------

def print_cluster_resources() -> dict[str, Any]:
    """Pretty-print the Ray cluster's CPU/GPU resources and return them."""
    import ray

    res = ray.cluster_resources()
    n_gpus = int(res.get("GPU", 0))
    n_cpus = int(res.get("CPU", 0))
    print(f"Ray cluster: {n_gpus} GPU(s), {n_cpus} CPU(s)")
    accel = [k.split(":", 1)[1] for k in res if k.startswith("accelerator_type:")]
    if accel:
        print(f"Accelerator type(s): {', '.join(accel)}")
    return res


# ---------------------------------------------------------------------------
# Model builders (vision examples used in 01/02/04)
# ---------------------------------------------------------------------------

def build_resnet18_mnist() -> torch.nn.Module:
    """ResNet-18 adapted for 1-channel (grayscale) MNIST, 10 classes.

    Used by the fast, robust fault-tolerance live-kill demo (notebook 04) and as
    an optional baseline in 01.
    """
    from torchvision.models import resnet18

    model = resnet18(num_classes=10)
    model.conv1 = torch.nn.Conv2d(
        in_channels=1, out_channels=64,
        kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False,
    )
    return model


class TokenizeText:
    """Stateful Ray Data transform. Loads a tokenizer once per actor.

    Notebook 01 builds this inline to teach the pattern. Notebooks 02 use it
    through `build_tokenized_text_dataset` so they can focus on training.
    """

    def __init__(self, model_name: str, seq_len: int):
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.seq_len = seq_len

    def __call__(self, batch):
        import numpy as np

        enc = self.tok(
            list(batch["text"]),
            padding="max_length", truncation=True, max_length=self.seq_len,
        )
        return {
            "input_ids": np.array(enc["input_ids"], dtype=np.int64),
            "attention_mask": np.array(enc["attention_mask"], dtype=np.int64),
        }


def build_tokenized_text_dataset(
    model_name: str,
    seq_len: int = 128,
    dataset_dir: str = "/mnt/cluster_storage/datasets/ag_news",
    actors: int = 2,
):
    """Return a Ray Dataset of tokenized text, ready for `get_dataset_shard`.

    Reads AG News from local parquet (staged from the tutorial's public S3
    mirror by the calling notebook), so this never touches the Hugging Face Hub.
    `model_name` is a local directory holding the tokenizer files.
    """
    import ray.data

    raw = ray.data.read_parquet(dataset_dir)
    return raw.map_batches(
        TokenizeText,
        fn_constructor_kwargs={"model_name": model_name, "seq_len": seq_len},
        batch_size=256,
        compute=ray.data.ActorPoolStrategy(size=actors),
    )


def build_causal_lm_dataloader(
    model_name: str,
    dp_rank: int,
    dp_size: int,
    seq_len: int = 128,
    batch_size: int = 1,
    data_dir: str = "/mnt/cluster_storage/datasets/wikitext-2-raw-v1",
    seed: int = 42,
):
    """A TP-aware causal-LM dataloader for notebook 03.

    The critical detail for tensor parallelism is that it shards by `dp_rank`
    and `dp_size`, NOT world rank and world size. Every tensor-parallel rank in
    the same data-parallel group must see the identical batch, or the gradients
    are wrong.

    Reads WikiText from local parquet (staged from the tutorial's public S3
    mirror by notebook 03) and loads the tokenizer from the local `model_name`
    directory, so this never touches the Hugging Face Hub.
    """
    import glob
    import os

    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    raw = load_dataset("parquet", data_files=files, split="train")
    tokenized = raw.map(
        lambda ex: tok(ex["text"], padding="max_length", max_length=seq_len, truncation=True),
        batched=True, remove_columns=raw.column_names,
    )
    tokenized = tokenized.filter(lambda e: sum(e["attention_mask"]) > 1)
    tokenized = tokenized.map(
        lambda ex: {
            "labels": [
                [tk if m == 1 else -100 for tk, m in zip(ids, mask)]
                for ids, mask in zip(ex["input_ids"], ex["attention_mask"])
            ]
        },
        batched=True,
    )
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # Shard by dp_rank / dp_size so TP ranks in a group share the same batch.
    sampler = DistributedSampler(
        tokenized, num_replicas=dp_size, rank=dp_rank, shuffle=True, seed=seed
    )
    return DataLoader(tokenized, batch_size=batch_size, sampler=sampler, drop_last=True)


def build_vit_cifar() -> torch.nn.Module:
    """A small Vision Transformer for 32x32 CIFAR-10 (10 classes).

    Used by the observability / profiling section (notebook 04): its clear
    transformer-block structure profiles nicely.
    """
    from torchvision.models import VisionTransformer

    return VisionTransformer(
        image_size=32, patch_size=4, num_layers=12, num_heads=8,
        hidden_dim=384, mlp_dim=768, num_classes=10,
    )
