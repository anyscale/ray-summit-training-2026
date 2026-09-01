import ray
from ray import serve
import torch
import pandas as pd
import requests
from starlette.requests import Request
import json
from sentence_transformers import SentenceTransformer
import textdistance
import numpy as np
from recommend import Recommend

@serve.deployment()
class Ingress:
    def __init__(self, recommender, search):
        self.recommender = recommender
        self.search = search

    async def __call__(self, request: Request):  # __call__ takes a Request object
        user = json.loads(await request.json()) # we will need to parse the JSON body of the request
        recommendations = (await self.recommender.get_recos.remote([user['id']]))[['item_id', 'name', 'desc', 'price']]
        result = { "recommendations" : json.loads(recommendations.to_json()) }
        if "query" in user:
            search_results = (await self.search.search.remote(user['query'], 5))[['item_id', 'name', 'desc', 'price']]
            result["search_results"] = json.loads(search_results.to_json())
        
        return json.dumps(result)
    
@serve.deployment()
class Recommender:
    def __init__(self, base_model_path: str, num_recos: int, num_users: int, num_products: int, database):
        self.recommend = Recommend(base_model_path, num_recos, num_users, num_products)
        self.db = database
    
    #async def recommend(self, user_ids):
    async def get_recos(self, user_ids):
        ref = self.db.users_for_ids.remote(user_ids)
        user_df = await ref
        recos = self.recommend.recommend_for_users(user_df.index.values)
        return await self.db.products_for_indices.remote(recos.flatten())
    
@serve.deployment()
class DatabaseFacade():
    def __init__(self, users, products):
        self.users = pd.read_json(users, lines=True)
        self.products = pd.read_parquet(products)
        
    def users_for_ids(self, ids):
        return self.users[self.users['id'].isin(ids)]
    
    def products_for_indices(self, idxs):
        return self.products.iloc[idxs]
    
    def all_products(self):
        return self.products

@serve.deployment()
class SemanticSearch():
    async def __init__(self, model, db):
        self.model = SentenceTransformer(model)
        self.db = db
        self.all_products_embeddings = (await self.db.all_products.remote())['desc_emb']
        
    async def search(self, query, matches):
        similarities = self.model.similarity(self.model.encode(query), self.all_products_embeddings)
        top_matches = similarities.flatten().topk(50).indices
        products = await self.db.products_for_indices.remote(np.array(top_matches))
        # rerank
        top_name_distances = torch.tensor([textdistance.lcsstr.similarity(query, prodname) for prodname in list(products['name'])]).topk(matches).indices
        results = products.iloc[np.array(top_name_distances)]
        
        return results
    
bound_db_facade_deployment = DatabaseFacade.bind('/mnt/cluster_storage/ecom/users.ndjson', '/mnt/cluster_storage/ecom/cat_with_embeddings')
bound_search = SemanticSearch.bind("/mnt/cluster_storage/ecom/hf_cache/models--google--embeddinggemma-300m/snapshots/57c266a740f537b4dc058e1b0cda161fd15afa75", bound_db_facade_deployment)
bound_rec_deployment = Recommender.bind('/mnt/cluster_storage/ecom/recommender/base_model/model.pt', 3, 1000, 1000, bound_db_facade_deployment)
bound_ingress = Ingress.bind(bound_rec_deployment, bound_search)

