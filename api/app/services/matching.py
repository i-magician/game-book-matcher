# Using sentence embeddings + cosine similarity to compare game 
# and book premises/themes to find which books are more thematically similar

import numpy as np 
from sentence_transformers import SentenceTransformer

def load_embedding_model(): 
    return SentenceTransformer("all-MiniLM-L6-v2")

def embed_text (model, text: str) -> np.ndarray:
    return model.encode(text)

def cosine_similarity( vec_a: np.ndarray, vec_b: np.ndarray ) -> float:
    dot_product = sum( a * b for a, b in zip( vec_a, vec_b))
    mag_a = sum( a ** 2 for a in vec_a ) ** 0.5
    mag_b = sum( b ** 2 for b in vec_b ) ** 0.5

    if mag_a == 0 or mag_b == 0:
        return 0
    return dot_product/(mag_a * mag_b)

def tag_similarity( tags_a: list [str], tags_b: list[str]) -> float:
    s1 = set(tags_a)
    s2 = set(tags_b)
    intersection = len( s1.intersection(s2))
    union = len(s1.union(s2))
    return intersection / union
