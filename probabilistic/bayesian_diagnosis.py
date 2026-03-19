"""
朴素贝叶斯诊断引擎

基于贝叶斯定理实现症状-疾病概率推断：
    P(疾病 | 症状) ∝ P(症状 | 疾病) × P(疾病)

实现要点：
- 均匀先验：每种疾病初始概率相等（1 / N）
- 朴素贝叶斯假设：各症状词条件独立
- Laplace 平滑：避免零概率问题，P(t|D) = (count+1) / (total+|V|)
- Log 空间计算：防止连乘下溢
- log-sum-exp 归一化：数值稳定地将 log 后验转为概率
- 预计算词频：构造时完成，推断时 O(|症状词| × |疾病数|)
"""

import json
import math
import jieba
from collections import Counter
from typing import List, Dict

# 过滤无意义的停用词
_STOPWORDS = {"，", "。", "、", " ", "\n", "\t", "的", "了", "和", "与", "或", "有", "无"}


class BayesianDiagnosisEngine:

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.diseases: List[Dict] = []
        self._disease_counters: List[Counter] = []   # 每个疾病的症状词频
        self._disease_totals: List[int] = []          # 每个疾病的症状词总数
        self.log_prior: float = 0.0                   # log P(疾病)，均匀先验
        self.vocab_size: int = 0                      # 全局词表大小（用于 Laplace 平滑）
        self._load_data()

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def _load_data(self):
        with open(self.data_path, "r", encoding="utf-8") as f:
            self.diseases = json.load(f)

        n = len(self.diseases)
        # 均匀先验：每种疾病概率相等
        self.log_prior = math.log(1.0 / n)

        # 预计算每种疾病的症状词频，同时构建全局词表
        all_tokens: set = set()
        for disease in self.diseases:
            tokens = self._tokenize(disease["symptoms"])
            counter = Counter(tokens)
            self._disease_counters.append(counter)
            self._disease_totals.append(max(sum(counter.values()), 1))
            all_tokens.update(counter.keys())

        # 词表大小用于 Laplace 平滑分母
        self.vocab_size = max(len(all_tokens), 1)

    # ------------------------------------------------------------------
    # 分词
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> List[str]:
        """中文分词，过滤停用词"""
        words = jieba.lcut(text)
        return [w for w in words if w not in _STOPWORDS and w.strip()]

    # ------------------------------------------------------------------
    # 似然估计
    # ------------------------------------------------------------------

    def _log_likelihood(self, user_tokens: List[str], disease_idx: int) -> float:
        """
        计算 log P(用户症状 | 疾病)

        朴素贝叶斯假设：各词独立，故：
            log P(S|D) = Σ_i log P(token_i | D)

        Laplace 平滑：
            P(token | D) = (match_count + 1) / (total_tokens_in_D + |V|)

        子串匹配：允许 "痒" 匹配 "瘙痒"，提升中文短词召回
        """
        counter = self._disease_counters[disease_idx]
        total = self._disease_totals[disease_idx]
        log_ll = 0.0

        for token in user_tokens:
            # 统计在该疾病症状词表中的匹配数（含子串匹配）
            matched = sum(
                cnt for d_tok, cnt in counter.items()
                if token in d_tok or d_tok in token
            )
            # Laplace 平滑后的条件概率
            prob = (matched + 1) / (total + self.vocab_size)
            log_ll += math.log(prob)

        return log_ll

    # ------------------------------------------------------------------
    # 推断接口
    # ------------------------------------------------------------------

    def predict(self, user_symptoms_text: str, top_k: int = 3) -> List[Dict]:
        """
        给定症状描述，返回 top_k 个最可能疾病及其后验概率。

        Args:
            user_symptoms_text: 用户描述的症状文本（中文）
            top_k: 返回结果数量

        Returns:
            [{"disease": str, "probability": float}, ...]
            按后验概率从高到低排序
        """
        user_tokens = self._tokenize(user_symptoms_text)
        if not user_tokens:
            return []

        # 计算每种疾病的 log 后验：log P(D|S) ∝ log P(S|D) + log P(D)
        scored = [
            (self._log_likelihood(user_tokens, i) + self.log_prior, d["disease"])
            for i, d in enumerate(self.diseases)
        ]
        scored.sort(reverse=True)

        # log-sum-exp 归一化：数值稳定地转换为概率
        log_scores = [s for s, _ in scored]
        max_log = log_scores[0]
        log_total = max_log + math.log(
            sum(math.exp(s - max_log) for s in log_scores)
        )

        return [
            {"disease": name, "probability": round(math.exp(log_s - log_total), 4)}
            for log_s, name in scored[:top_k]
        ]
