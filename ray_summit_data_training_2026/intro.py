import re
import os
import pickle
import tempfile
import numpy as np
import torch.nn as nn
import ray
import torch
import ray.train.torch
import boto3
from ray.train.torch import TorchTrainer
from ray.train import Checkpoint, ScalingConfig, RunConfig
from ray.tune.integration.ray_train import TuneReportCallback
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_and_combine(row):
    title = row["title"].lower()
    desc = row["description"].lower()
    # Strip punctuation
    title = re.sub(r"[^\w\s]", "", title)
    desc = re.sub(r"[^\w\s]", "", desc)
    row["text"] = title + " " + desc
    return row

class VectorizeAndEncode:
    def __init__(self):
        s3 = boto3.client('s3')
        bucket_name = 'anyscale-public-materials-use2'
        vec_object_key = 'ecom/intro/vectorizer.pickle'
        idx_object_key = 'ecom/intro/category_index_map.pickle'

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            s3.download_fileobj(bucket_name, vec_object_key, temp_file)
            self.vectorizer = pickle.load(open(temp_file.name, "rb"))
    
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            s3.download_fileobj(bucket_name, idx_object_key, temp_file)
            temp_file.seek(0)
            self.cat_to_idx = pickle.load(open(temp_file.name, "rb"))
            
    def __call__(self, batch):        
        tfidf = self.vectorizer.transform(batch["text"]).astype(np.float32).todense() # TF-IDF: sparse matrix → dense float32 array        
        labels = [self.cat_to_idx[val] for val in batch['category']] # Label encode the category column
        
        return {
            "features": np.array(tfidf), # each element is a 1-D array of length 128
            "label": np.array(labels),
        }

# INPUT_DIM = 128 # matches TF-IDF max_features
# NUM_CLASSES = 6 # len(cat_to_idx)

class ProductClassifier(nn.Module):
    def __init__(self, input_dim=128, num_classes=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_loop(config):
    from ray.train import get_dataset_shard
    
    model = ProductClassifier(128, 6)
    model = ray.train.torch.prepare_model(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss()

    train_shard = get_dataset_shard("train")

    for epoch in range(5):
        total_loss = 0.0
        num_batches = 0

        for batch in train_shard.iter_torch_batches(batch_size=64):
            features = batch["features"].float()
            labels = batch["label"].long()

            outputs = model(features)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:            
            torch.save(model.module.state_dict(), os.path.join(temp_checkpoint_dir, "model.pt"))
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            ray.train.report({"loss": avg_loss, "epoch": epoch}, checkpoint=checkpoint)

def train_driver(config, dataset=None):
    """Tune calls this once per trial with sampled hyperparameters."""
    
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop,
        train_loop_config={
            "input_dim": 128,
            "num_classes": 6,
            "epochs": 4,
            "lr": config["lr"], # <---------
        },
        scaling_config=ScalingConfig(num_workers=2),
        datasets={"train": dataset},
        run_config=RunConfig(storage_path='/mnt/cluster_storage/',
                             callbacks=[TuneReportCallback()]) # Bridge Train metrics → Tune
    )
    trainer.fit()
