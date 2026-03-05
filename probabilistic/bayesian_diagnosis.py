import json
import os
import math
import jieba

class BayesianDiagnosisEngine:
    def __init__(self, data_path):
        self.data_path = data_path
        self.diseases = []
        self._load_data()

    def _load_data(self):
        with open(self.data_path, "r", encoding="utf-8") as f:
            self.diseases = json.load(f)

        # 先验概率
        self.prior = 1 / len(self.diseases)

    # 中文分词
    def _tokenize(self, text):
        return list(set(jieba.lcut(text)))

    def predict(self, user_symptoms, top_k=3):

        user_symptoms = self._tokenize(user_symptoms)

        results = []

        for disease in self.diseases:

            disease_symptoms = self._tokenize(disease["symptoms"])

            score = 0

            for us in user_symptoms:

                if any(us in ds or ds in us for ds in disease_symptoms):
                    score += 2   # 强匹配
                else:
                    score -= 1   # 不匹配

            results.append({
                "disease": disease["disease"],
                "score": score
            })

        results.sort(key=lambda x: x["score"], reverse=True)

        # 转概率
        scores = [math.exp(r["score"]) for r in results]
        total = sum(scores)

        for i in range(len(results)):
            results[i]["probability"] = round(scores[i] / total, 4)

        return results[:top_k]