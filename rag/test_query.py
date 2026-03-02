import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt


# 读取数据
with open("../data/medical_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 加载索引
index = faiss.read_index("medical_index.faiss")

model = SentenceTransformer('all-MiniLM-L6-v2')

# 用户输入
query = "皮肤很痒还有红肿"
query_vector = model.encode([query])

D, I = index.search(np.array(query_vector), k=3)

print("RAG检索结果：")
retrieved = []
for idx in I[0]:
    print(data[idx]["disease"])
    retrieved.append(data[idx])

# ===== 数学概率评估模块 =====

def calculate_probability(query, disease_symptoms):
    total_chars = 0
    matched_chars = 0

    for symptom in disease_symptoms.split():
        for char in symptom:
            total_chars += 1
            if char in query:
                matched_chars += 1

    return matched_chars / total_chars if total_chars > 0 else 0

print("\n概率评估结果：")

diseases = []
probs = []

for item in retrieved:
    prob = calculate_probability(query, item["symptoms"])
    diseases.append(item["disease"])
    probs.append(prob)
    print(f"{item['disease']} : {round(prob,2)}")

# 可视化
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.bar(diseases, probs)
plt.title("疾病概率分布")
plt.ylim(0,1)
plt.show()