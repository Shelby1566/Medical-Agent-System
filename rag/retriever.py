import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "faiss_index.pkl")

# 加载 embedding 模型
model = SentenceTransformer("BAAI/bge-small-zh")
# 加载索引
with open(INDEX_PATH, "rb") as f:
    index, documents = pickle.load(f)

def retrieve(query, top_k=2):
    query_vector = model.encode([query])
    query_vector = np.array(query_vector).astype("float32")

    distances, indices = index.search(query_vector, top_k)

    results = [documents[i] for i in indices[0]]

    return "\n\n".join(results)
