import json
import os
import math
from collections import defaultdict

class BayesianDiagnosisEngine:
    def __init__(self, data_path):
        self.data_path = data_path
        self.diseases = []
        self._load_data()

    def _load_data(self):
        with open(self.data_path, "r", encoding="utf-8") as f:
            self.diseases = json.load(f)

        # 计算先验概率（默认均匀分布）
        self.prior = 1 / len(self.diseases)

    def _tokenize(self, text):
        # 简单按空格、顿号、逗号分割
        separators = [" ", "、", ",", "，"]
        for sep in separators:
            text = text.replace(sep, " ")
        return list(set(text.split()))

    def predict(self, user_symptoms, top_k=3):
        user_symptoms = self._tokenize(user_symptoms)

        results = []

        for disease in self.diseases:
            disease_symptoms = self._tokenize(disease["symptoms"])

            # 使用对数避免下溢
            log_prob = math.log(self.prior)

            for symptom in user_symptoms:
                if symptom in disease_symptoms:
                    likelihood = 0.8   # 出现症状的条件概率
                else:
                    likelihood = 0.2   # 未出现的平滑概率

                log_prob += math.log(likelihood)

            results.append({
                "disease": disease["disease"],
                "score": log_prob
            })

        # 按概率排序
        results.sort(key=lambda x: x["score"], reverse=True)

        # 转换成归一化概率
        scores = [math.exp(r["score"]) for r in results]
        total = sum(scores)

        for i in range(len(results)):
            results[i]["probability"] = round(scores[i] / total, 4)

        return results[:top_k]
