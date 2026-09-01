# Multimodal Data Pipelines for AI Systems

***(SHORT OUTLINE)***

This course introduces Ray and the Anyscale platform through a hands-on, end-to-end e-commerce scenario. We move from a high-level overview of the Ray ecosystem to a deep, practical treatment of large-scale data processing and multimodal inference with Ray Data.

* Introduction: Ray, Anyscale, and the AI Libraries  
* What is Ray and Anyscale  
* End-to-end demo — product category prediction  
* A quick look at Ray Core  
* Transform and Ingest an E-Commerce Catalog with Ray Data  
* Ingest and basic transformation  
* Filtering and the Ray Data expression language  
* Generating embeddings for semantic search  
* Performance, parallelism, and scaling  
* Generating catalog copy with a multimodal model  
* AI-assisted development of Ray pipelines  
* Accelerated inference with Ray \+ vLLM  
* Demo: Real-world operationalizing with Ray and Anyscale Jobs

**Planned Compute:**

```
compute_config:   
  head_node:  
    instance_type: m5.2xlarge  
  worker_nodes:  
    - instance_type: g5.2xlarge  
      min_nodes: 2  
      max_nodes: 2
```

**Minimum Compute:**
A10G or better acceleration, 48GB total GPU RAM

***(LONG OUTLINE)***

## 1 — Introduction: Ray, Anyscale, and the AI Libraries

### 1.1 What is Ray and Anyscale

- Ray as an open-source framework for high-performance, resilient, scale-out computation on heterogeneous hardware  
- The distributed scheduler: stateless functions ("tasks") and long-running stateful processes ("actors")  
- Key capabilities: dependency tracking via task graphs, resource-aware data movement, mixed/fractional/custom resource requirements, the object store, and fault tolerance via the GCS  
- The Ray AI Libraries as high-level APIs layered on Ray Core  
- Anyscale as production-ready Ray: multi-node IDE and observability, an optimized runtime, the cluster controller, and support/services

### 1.2 End-to-end demo — product category prediction

A single example using every major Ray AI Library together (hiding much code for brevity):

- Load and process data with Ray Data   
- Train/test split  
- Train a model with Ray Train  
- Tune hyperparameters with Ray Tune  
- Batch inference with Ray Data  
- Online prediction with Ray Serve

### 1.3 A quick look at Ray Core

- `@ray.remote` and `ray.get`; the meaning and purpose of `ObjectRef` handles  
- Tasks launching other tasks (nested parallelism)  
- Building task graphs by passing `ObjectRef`s (and/or concrete values) as arguments, with automatic dependency resolution  
- Ray Actors — stateful class instances that persist in the cluster,

## 2 — Transform and Ingest E-Commerce Catalog with Ray Data

### 2.1 Ingest and basic transformation

- Reading tabular Parquet data and inspecting it (`count`, `take`, `take_batch`)  
- Understanding the record schema (manufacturer/vendor IDs, category, name, description, price, parent-variant linkage)  
- Datasets as streaming, lazy abstractions: chained transformations produce new Datasets cheaply, and a pipeline must terminate in a write/consume step  
- Anyscale storage scopes — shared org storage, cluster storage, per-user, machine-local, and cloud blob storage as best-practice defaults

### 2.2 Filtering and the Ray Data expression language

- Filtering rows (e.g., by category) with a lambda  
- Why opaque lambdas limit optimization, and how Ray Data's query optimizer/planner (logical and physical optimizations, as of 2026\) needs legible operands  
- The expression language (`col(...)`) as the optimizer-friendly alternative  
- Columnar formats and predicate pushdown

### 2.3 Generating embeddings for semantic search

- Starting from non-Ray "sample code" (a `SentenceTransformer` embedding model) and adapting it for Ray  
- The core batch-inference pattern: a plain Python class holding model state in `__init__` and transforming batches in `__call__`, applied via `map_batches` with `fn_constructor_args` and `fn_args`  
- **Developer workflow tips:** use `limit` plus `materialize` to compute and cache a small sample for fast iteration

### 2.4 Performance, parallelism, and scaling

- Ray Data **blocks** as the core streaming unit, and how block count drives Actor parallelism  
- Why a small dataset runs on a single Actor, and using `repartition` to increase parallelism  
- Observing actors and progress in the Ray Dashboard  
- Moving from CPU to GPU inference: modifying the actor for `cuda`, setting `num_gpus`, choosing `batch_size` for GPU memory, and supplying an `ActorPoolStrategy` (GPU actors don't autoscale automatically)

### 2.5 Generating catalog copy with a multimodal model

- Goal: richer, SEO-friendly product descriptions from each item's text **and** image  
- Validating a multimodal "hello world" (`image-text-to-text`, Gemma 3\) before scaling  
- Prototyping inside a Ray Actor first — to confirm accelerator use, iterate on prompts/output format, and gauge GPU memory for batch sizing  
- Testing against real catalog products and images; releasing the actor's GPU with `ray.kill`  
- **Suggested exercise:** refactor to batch inputs and find the maximum reliable batch size

### 2.6 AI-assisted development of Ray pipelines

- Using an AI coding assistant to convert the Hugging Face "hello world" into Ray Data code, given the snippet, the dataset schema, and a task description  
- Reviewing, refactoring, and testing the generated pipeline (removing scaffolding, using the known-good system prompt, passing images by path, implementing batching)  
- Iterating safely on a tiny sample dataset before a full run  
- Catching and fixing API drift with AI help — e.g., the deprecated `concurrency` argument replaced by `compute=ActorPoolStrategy(...)`  
- Theme throughout: pairing AI productivity with human verification, statistical evals, and output checks

### 2.7 Accelerated inference with Ray \+ vLLM

- What is vLLM  
- What is ray.data.llm  
- Architecture/pattern change when using ray.data.llm  
- Implementing the code change  
- Performance check  
- Demo: Real-world operationalizing with Ray and Anyscale Jobs

### 2.8 Demo: Real-world operationalizing with Ray and Anyscale Jobs

- Instructor demo:  
  - Linear speedup using a larger cluster  
  - Configuring code and YAML for a Job  
  - Running jobs using the CLI  
  - Observing the running and completed job

