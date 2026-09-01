from transformers import pipeline
import ray
import torch
import os

MODEL_NAME = "/mnt/shared_storage/course_assets/ecom/hf_cache/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767"
IMAGE_DIR = "/mnt/shared_storage/course_assets/ecom/catalog_images/"
INPUT_DIR = "/mnt/user_storage/cat_with_embeddings/" # <- needs to exist outside of the job's cluster
OUTPUT_PATH = "/mnt/user_storage/cat_with_embed_and_extended_desc.parquet" # <- we want this to exist after the job
BATCH_SIZE = 8 

system_prompt = '''You are a helpful assistant. Given a product name and description, and an image of that product, 
please create a more descriptive and attractive blurb for the product, 
capturing elements from the image and suitable for use in an ecommerce website where that product is for sale.
Output a single suggestion or option, and do not include any additional conversational language or discussion.
'''

class VLMPredictor:
    def __init__(self):
        self.pipe = pipeline("image-text-to-text", model=MODEL_NAME, device="cuda:0")

    def __call__(self, batch: dict) -> dict:
        results = []
        messages = []

        for item_id, desc in zip(batch["item_id"], batch["desc"]):
            image_path = os.path.join(IMAGE_DIR, f"{item_id}.png")
            content = []
            content.append({"type": "image", "path": image_path})
            content.append({"type": "text", "text": desc})

            messages.append([
                { "role": "system", "content": [{"type": "text", "text": system_prompt}] },
                {"role": "user", "content": content}
            ])
            
        outputs = self.pipe(text=messages, max_new_tokens=1024, batch_size=len(messages))          
        batch["cat_desc"] = [out[0]["generated_text"][-1]["content"] for out in outputs]
        return batch

ray.init()
GPUs = int(ray.cluster_resources()['GPU'])

ray.data.read_parquet(INPUT_DIR).repartition(2*GPUs
    ).map_batches(
        VLMPredictor,
        batch_size=BATCH_SIZE,
        compute=ray.data.ActorPoolStrategy(size=GPUs),
        num_gpus=1,
    ).write_parquet(OUTPUT_PATH, mode=ray.data.SaveMode.OVERWRITE)

print(f'Catalog with extended descriptions added at {OUTPUT_PATH}')
