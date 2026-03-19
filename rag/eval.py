"""
模块评测脚本

对贝叶斯推断模块和 RAG 检索模块分别进行定量评测。

评测方法：
  从疾病数据集中等间隔采样 N 个疾病，
  取每个疾病的"部分症状"作为模拟用户输入（模拟患者描述不完整的情况），
  检验系统能否在 Top-K 结果中召回该疾病。

评测指标：
  - 贝叶斯 Top-1 准确率：预测第一名是否命中
  - 贝叶斯 Top-3 准确率：Top-3 中是否命中
  - RAG Top-3 召回率：检索结果中是否包含该疾病名称

运行方式：
    python rag/eval.py
"""

import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from probabilistic.bayesian_diagnosis import BayesianDiagnosisEngine
from rag.retriever import retrieve

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "medical_data.json")

# ── 评测参数 ──────────────────────────────────────────────────────────
N_SAMPLES = 20          # 采样数量
SYMPTOM_RATIO = 0.5     # 使用症状词的比例（模拟输入不完整）


def build_test_cases(data_path: str, n: int, ratio: float) -> list:
    """
    从数据集中等间隔采样构建测试用例。

    每个用例取该疾病症状词的前 ratio 比例作为输入，
    检验系统能否从不完整描述中召回正确疾病。
    """
    with open(data_path, "r", encoding="utf-8") as f:
        diseases = json.load(f)

    step = max(1, len(diseases) // n)
    sampled = diseases[::step][:n]

    test_cases = []
    for disease in sampled:
        words = disease["symptoms"].split()
        # 取前 ratio 比例的症状词，至少保留 2 个
        partial_count = max(2, int(len(words) * ratio))
        partial_symptoms = " ".join(words[:partial_count])
        test_cases.append({
            "input":    partial_symptoms,
            "expected": disease["disease"],
            "full":     disease["symptoms"],
        })

    return test_cases


def evaluate_bayesian(engine: BayesianDiagnosisEngine, test_cases: list) -> dict:
    """评测贝叶斯模块的 Top-1 和 Top-3 准确率。"""
    top1_hits = 0
    top3_hits = 0
    details = []

    for case in test_cases:
        preds = engine.predict(case["input"], top_k=3)
        pred_names = [p["disease"] for p in preds]
        top1 = pred_names[0] if pred_names else ""

        hit1 = (top1 == case["expected"])
        hit3 = (case["expected"] in pred_names)

        top1_hits += int(hit1)
        top3_hits += int(hit3)

        details.append({
            "输入症状":  case["input"],
            "预期疾病":  case["expected"],
            "Top-1 预测": top1,
            "Top-3 命中": "✅" if hit3 else "❌",
            "Top-3 结果": " / ".join(pred_names),
        })

    n = len(test_cases)
    return {
        "top1_acc": top1_hits / n,
        "top3_acc": top3_hits / n,
        "details": details,
    }


def evaluate_rag(test_cases: list) -> dict:
    """评测 RAG 检索模块的 Top-3 召回率（疾病名是否出现在检索文本中）。"""
    hits = 0
    details = []

    for case in test_cases:
        retrieved_text = retrieve(case["input"], top_k=3)
        hit = case["expected"] in retrieved_text
        hits += int(hit)

        details.append({
            "输入症状":  case["input"],
            "预期疾病":  case["expected"],
            "RAG 召回": "✅" if hit else "❌",
        })

    return {
        "recall": hits / len(test_cases),
        "details": details,
    }


def print_table(headers: list, rows: list):
    """打印简易文本表格。"""
    col_widths = [max(len(str(r[h])) for r in rows + [{h: h}]) for h in headers]
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    separator = "  ".join("-" * w for w in col_widths)
    print(fmt.format(*headers))
    print(separator)
    for row in rows:
        print(fmt.format(*[str(row[h]) for h in headers]))


def main():
    print("=" * 60)
    print("  医疗诊断系统 - 模块评测")
    print(f"  采样数量：{N_SAMPLES}  症状完整度：{SYMPTOM_RATIO:.0%}")
    print("=" * 60)

    print("\n构建测试集...")
    test_cases = build_test_cases(DATA_PATH, N_SAMPLES, SYMPTOM_RATIO)
    print(f"已生成 {len(test_cases)} 个测试用例\n")

    # ── 贝叶斯评测 ────────────────────────────────────────────
    print("─" * 60)
    print("【贝叶斯推断模块评测】")
    print("─" * 60)
    engine = BayesianDiagnosisEngine(DATA_PATH)
    bayes_result = evaluate_bayesian(engine, test_cases)

    print_table(
        ["输入症状", "预期疾病", "Top-1 预测", "Top-3 命中"],
        [{k: v for k, v in row.items() if k != "Top-3 结果"}
         for row in bayes_result["details"]],
    )

    print(f"\n  Top-1 准确率：{bayes_result['top1_acc']:.1%}")
    print(f"  Top-3 准确率：{bayes_result['top3_acc']:.1%}")

    # ── RAG 评测 ──────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("【RAG 检索模块评测】")
    print("─" * 60)
    rag_result = evaluate_rag(test_cases)

    print_table(
        ["输入症状", "预期疾病", "RAG 召回"],
        rag_result["details"],
    )

    print(f"\n  RAG Top-3 召回率：{rag_result['recall']:.1%}")

    # ── 综合总结 ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  评测总结")
    print("=" * 60)
    print(f"  测试样本数：    {len(test_cases)}")
    print(f"  贝叶斯 Top-1：  {bayes_result['top1_acc']:.1%}")
    print(f"  贝叶斯 Top-3：  {bayes_result['top3_acc']:.1%}")
    print(f"  RAG 召回率：    {rag_result['recall']:.1%}")
    print("=" * 60)
    print("\n注：评测使用部分症状作为输入（完整症状的前50%），")
    print("    模拟真实场景中患者描述不完整的情况。\n")


if __name__ == "__main__":
    main()
