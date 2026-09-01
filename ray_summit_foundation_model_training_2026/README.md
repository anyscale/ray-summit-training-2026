# Foundation Model Distributed Training with Ray

Build and optimize large-scale distributed training systems for foundation
models on **Ray + Anyscale**. This course walks the full path from a single-GPU
training loop to multi-GPU sharded and tensor-parallel training, with
checkpointing, fault recovery, and deep observability along the way. You run
every step yourself on a 4-GPU cluster.

> **One Ray surface. Two GPUs or two hundred. Change the config, not the code.**

The models (gpt2, Qwen2.5-0.5B, ResNet, ViT) are just the workload. The lesson
is how a small set of Ray primitives (`TorchTrainer`, `prepare_model`,
`ray.train.report`, `ScalingConfig`, `FailureConfig`, and Ray Data) carry
unchanged from plain data-parallel training, to FSDP and DeepSpeed sharding, to
2D tensor-plus-data parallelism, to fault-tolerant production runs. Swap in your
own foundation model and the orchestration code barely changes.

---

## The arc

```
  WHY            scaling laws, the memory wall, the five ways to split work
   │
   ▼
  RAY FOUNDATIONS    one GPU  ->  two GPUs with prepare_model + Ray Data
   │
   ▼
  MEMORY & SHARDING  the model does not fit  ->  shard it (FSDP, DeepSpeed ZeRO)
   │
   ▼
  2D PARALLELISM     the tensors do not fit  ->  split them (TP x DP)
   │                 pipeline and expert parallelism as the next axes
   ▼
  PRODUCTION         it fails and you fly blind  ->  fault tolerance + observability
   │
   ▼
  PUTTING IT TOGETHER   considerations when combining everything
```

---

## Course map

| # | Notebook | What you learn | Ray surface |
|---|----------|----------------|-------------|
| pre | `prerequisite_00_ray_data.ipynb` | Streaming data loading, transforms, batch inference | **Ray Data** |
| 00 | `00_scaling_imperative.ipynb` | Why we scale, the memory wall, the five parallelism axes | framing |
| 01 | `01_ray_foundations.ipynb` | Single-GPU to distributed DDP with one config change | **Ray Train + Data** |
| 02 | `02_memory_and_sharding.ipynb` | Activation checkpointing, FSDP, DeepSpeed ZeRO, memory profiling | **Ray Train + FSDP / DeepSpeed** |
| 03 | `03_2d_parallelism.ipynb` | Tensor + data parallelism (DTensor, AutoTP), pipeline and expert concepts | **Ray Train orchestrating TP x DP** |
| 04 | `04_fault_tolerance_and_observability.ipynb` | Checkpointing, a live worker kill, elastic training, dashboards and profiling | **Ray fault tolerance + Anyscale observability** |
| 05 | `05_putting_it_together.ipynb` | 5D parallelism, the bandwidth hierarchy | synthesis |

**Outline coverage.** Data, tensor, pipeline, and expert parallelism (00, 03,
05). FSDP and DeepSpeed architectures (02, 03). Multi-node GPU scheduling and
orchestration (01). Checkpointing and fault recovery (04). Distributed data
sharding and ingestion with Ray Data (prereq, 01). Training throughput and
utilization optimization (04).

The notebooks are designed to be read in order and they cross-reference each
other. They ship ready to run on a 4-GPU cluster. Every notebook executes top to
bottom at smoke scale, so you run each cell yourself and watch it work.

---

## Shared modules

| File | Role |
|------|------|
| `common/utils.py` | Mixed-precision selection, the standard Ray runtime environment, model builders, cluster printer |
| `common/kill_train_worker.py` | Utility that SIGKILLs a live training worker to demonstrate fault recovery (notebook 04) |

---

## Prerequisites

### Cluster

A **2x NVIDIA T4 Anyscale cluster** running the image below. The head node is
CPU-only. The two T4 GPUs live on `2xg4dn.2xlarge` worker nodes. Two GPUs is the constant. The same code runs whether those two GPUs
sit on one node or two, because placement is Ray's job, not yours.

The T4 is a Turing GPU. It has fp16 tensor cores but no native bf16, so every
mixed-precision path in this course selects fp16. The GPUs connect over PCIe,
not NVLink, which is why we keep tensor-parallel degree small. Both points are
real lessons the notebooks call out.

### Dependencies

Installed once with plain `pip`, which an Anyscale workspace registers on every
node in the cluster.

```bash
pip install -r requirements.txt
```


### Storage

This is an Anyscale workspace, so all nodes share `/mnt/cluster_storage`. Every
checkpoint, profiler trace, and result in this course is written there so all
workers can read and write it.

### Models and datasets (no Hugging Face token needed)

**Nothing in this course reaches the Hugging Face Hub at runtime.** Every model
(gpt2, Qwen2.5-0.5B) and dataset (AG News, WikiText-2, MNIST) is mirrored to a
public S3 bucket:

```
s3://anyscale-public-materials/ray_summit_foundation_model_training_2026/
├── models/gpt2/                     # full snapshot (config + tokenizer + safetensors)
├── models/qwen2.5-0.5b/
├── datasets/ag_news/                # parquet (train slice)
├── datasets/wikitext-2-raw-v1/      # parquet (train slice)
└── datasets/mnist/                  # torchvision raw/ layout
```

Each notebook sets `HF_HUB_OFFLINE=1` / `HF_DATASETS_OFFLINE=1` (so an accidental
repo-id load raises instead of hitting the Hub), then stages what it needs into
shared `/mnt/cluster_storage` with an anonymous, unsigned read:

```bash
aws s3 sync s3://anyscale-public-materials/ray_summit_foundation_model_training_2026/models/gpt2 \
            /mnt/cluster_storage/models/gpt2 --no-sign-request
```

and loads from the local path (`from_pretrained("/mnt/cluster_storage/models/gpt2")`,
`ray.data.read_parquet(...)`). No token, no credentials, no license gate — so
hundreds of attendees can run the course at once and **Hugging Face cannot
throttle the event**. Reads are anonymous; each cluster only writes to its own
`/mnt/cluster_storage`.

**Presenters — re-seeding the mirror.** The mirror was seeded once by downloading
the (ungated) models and dataset slices from Hugging Face and uploading them to
the bucket. To rebuild it, download `gpt2` and `Qwen/Qwen2.5-0.5B` with
`huggingface-cli download`, export `ag_news[:2%]` and `wikitext-2-raw-v1[:3%]` to
parquet, download MNIST with `torchvision`, then `aws s3 sync` each into the
prefix above. Writing to the bucket requires credentials for the
`anyscale-public-materials` account.

### Tested configuration

| Component | Version |
|-----------|---------|
| Anyscale image | `anyscale/ray:2.55.1-py311-cu128` |
| Ray | 2.55.1 |
| Python | 3.11 |
| PyTorch | 2.9.1 + CUDA 12.8 |
| transformers | 4.48.0 |
| datasets | 2.21.0 |
| deepspeed | 0.18.9 |
| accelerate | 1.3.0 |


---

## A note on scope

Every run here is at smoke scale, with small step counts so it finishes in
minutes. The goal is to teach the mechanics and the orchestration, not to chase
a low loss. Pipeline and expert parallelism are presented as concepts with
diagrams, because they cannot train meaningfully on two T4 GPUs. The same code
patterns scale to production by changing config, not logic.
