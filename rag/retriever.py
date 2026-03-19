"""
RAG 检索模块

基于 FAISS 向量索引 + BGE-small-zh 语义嵌入，
对用户查询执行近似最近邻检索，返回最相关的疾病知识条目。

懒加载设计：模型和索引在首次调用 retrieve() 时才加载。
"""

import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from typing import List

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "faiss_index.pkl")

_model = None
_index = None
_documents: List[str] = []


def _load_resources():
    """懒加载：首次检索时加载嵌入模型和 FAISS 索引。"""
    global _model, _index, _documents
    if _model is None:
        _model = SentenceTransformer("BAAI/bge-small-zh")
    if _index is None:
        with open(INDEX_PATH, "rb") as f:
            _index, _documents = pickle.load(f)


def retrieve(query: str, top_k: int = 3) -> str:
    """
    语义检索与查询最相关的疾病知识条目。

    Args:
        query:  检索查询（症状描述文本）
        top_k:  返回条目数量

    Returns:
        拼接后的检索结果字符串（各条目间以空行分隔）
    """
    _load_resources()
    query_vector = np.array(_model.encode([query])).astype("float32")
    _, indices = _index.search(query_vector, top_k)
    results = [_documents[i] for i in indices[0]]
    return "\n\n".join(results)
