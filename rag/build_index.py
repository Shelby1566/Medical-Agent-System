import json
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../data/medical_data.json")
INDEX_PATH = os.path.join(BASE_DIR, "faiss_index.pkl")

# 1️⃣ 读取 JSON
with open(DATA_PATH, "r", encoding="utf-8") as f:
    medical_data = json.load(f)

# 2️⃣ 组织文本（把每条疾病转成可向量化文本）
documents = []
for item in medical_data:
    text = f"""
    疾病名称: {item['disease']}
    典型症状: {item['symptoms']}
    疾病描述: {item['description']}
    """
    documents.append(text.strip())

# 3️⃣ 加载 embedding 模型
model = SentenceTransformer("BAAI/bge-small-zh")

# 4️⃣ 生成向量
embeddings = model.encode(documents)
embeddings = np.array(embeddings).astype("float32")

# 5️⃣ 创建 FAISS 索引
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# 6️⃣ 保存 index + 原始文本
with open(INDEX_PATH, "wb") as f:
    pickle.dump((index, documents), f)

print("✅ FAISS 索引构建完成")