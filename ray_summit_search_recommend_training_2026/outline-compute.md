# Search & Recommendation Systems for AI Commerce

***(SHORT OUTLINE)***

This course teaches the batch inference, online serving and model updating (training) sides of a production recommender and semantic-search system, built on Ray for an e-commerce scenario. Learners work with a pre-trained two-tower recommender model and a product catalog with description embeddings, and they progress from running inference to periodically re-training the model on fresh interaction data.

* Framing the inference problems: batch use cases, online use cases  
* Implementing full batch pipeline  
* Online services with Ray Serve, starting simple: a single Deployment   
* Refactoring toward separation of concerns into composed deployments  
* Composing deployments  
* Adding a SemanticSearch deployment  
* Updating Ingress to route  
* Deploying a Ray Serve service via the CLI  
* Demo: Productionizing with Anyscale Services  
* Re-train (Update) the Recommender with Ray Train  
* Baseline PyTorch training and Minimal port to Ray Train  
* Integrating a Ray Data pipeline for training data  
* Toward scheduled re-training: wrapping the training program in a Ray or Anyscale Job

---

**Planned Compute:**

```
compute_config:   
  head_node:  
    instance_type: m5.2xlarge  
  worker_nodes:  
    - instance_type: m5.2xlarge  
      min_nodes: 2  
      max_nodes: 2
```

***(LONG OUTLINE)***

## 1 — Model Inference for Batch and Online Use Cases (Ray Data & Ray Serve)

### 1.1 Framing the inference problems

- Batch use case: for every user in a dataset, generate product recommendations, compose a personalized email, and write to storage / a mail queue / a database  
- Online use cases: generate a recommendation for a live user session, and combine recommendation with product search when a query is present  
- Working from a pre-trained recommender model snapshot rather than training from scratch

### 1.2 Batch inference with a Ray Data pipeline

- Reading a synthetic user dataset and the troubleshooting   
- Designing a composable pipeline: read users → map GUIDs to integer indices → infer recommendations → look up product details → generate emails → enqueue for sending  
- The named Ray Actor as a database facade and retrieving it anywhere via `ray.get_actor`  
- Stateless `map_batches` functions vs. the stateful actor-class pattern for the model  
- Loading the **two-tower recommender** inside a `Recommend` actor class and producing top-k recommendations per user  
- Implementing full pipeline and simulating an email-send queue as a side-effecting `map_batches` stage

### 1.3 Online services with Ray Serve

- Starting simple: a single `Deployment`   
- Binding and running a deployment, calling it via a handle, and via HTTP  
- Refactoring toward separation of concerns into composed deployments: `Ingress`, `Recommender`, and a `DatabaseFacade` deployment  
- The async serving pattern: `await` on deployment handles instead of `ray.get`  
- Composing deployments in dependency order and wiring them together  
- Adding a `SemanticSearch` deployment that embeds the query, scores against precomputed product description embeddings, and re-ranks by string distance  
- Updating `Ingress` to route

### 1.4 Deploying via the CLI

- Inspecting the refactored modules (`recommend.py`, `search_and_recommend.py`) with attention to runtime file paths, imported code/working dir, environment variables, and namespacing in a distributed runtime  
- `serve build` to generate a `serve_config.yaml`, then editing it  
- `serve deploy` from the CLI

### 1.5 Demo: Productionizing with Anyscale Services

- Instructor demo:  
  - Extending the Serve config with Anyscale entries: service name, image URI, compute config, cloud, working dir, and query auth token  
  - Best practices: a global working dir that includes the service script, and no dependencies on the development workspace  
  - `anyscale service deploy`, checking status, and querying the authenticated endpoint  
  - **Canary rollouts** — deploying a new version alongside the old, gradual traffic shifting, version-routing headers for testing, and rollback; terminating the service

---

## 2 — Re-train (Update) the Recommender with Ray Train

### 2.1 Baseline PyTorch training

- The `TwoTower` model (user/item embeddings, normalized encodings, in-batch negatives via a `[B,B]` logits matrix)  
- A plain local training loop

### 2.2 Minimal port to Ray Train

- Converting the loop for distributed Torch DDP with `ray.train.torch.prepare_model` (implicit device detection, data/model movement, and DDP wrapping)  
- Orchestrating with `TorchTrainer`, `ScalingConfig` (workers, `use_gpu`), and a `RunConfig` storage path; calling `.fit()`

### 2.3 Configuration, checkpointing, and metrics

- Parametrizing the loop with `train_loop_config`   
- Checkpointing and reporting via Ray Train APIs  
- Unwrapping `model.module` for DDP, and saving only on global rank 0

### 2.4 Integrating a Ray Data pipeline for training data

- Why combine Ray Data with Ray Train: parallelized/accelerated preprocessing, separate hardware for data vs. training, and optimized batch delivery into workers  
- Building the preprocessing pipeline  
- Consuming data inside the worker: `get_dataset_shard` for per-worker shards and `iter_torch_batches` (batch size, format/dtype, prefetch, shuffle)  
- Supplying the pipeline to `TorchTrainer` via `datasets={'train': ...}`

### 2.5 Epochs, streaming vs. materialized data

- Refactoring into an epoch/batch loop, computing per-worker batch size from the global batch size and world size, with reporting/checkpointing per epoch  
- Understanding that the Dataset re-executes every epoch by default  
- Caching with `materialize` and observing the difference in the logs

### 2.6 Toward scheduled re-training

- Wrapping the training program in a Ray or Anyscale **Job** to regularly consume new data, fine-tune the existing model, and publish fresh checkpoints to a known location

