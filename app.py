"""
医疗多模态智能咨询系统 - Streamlit 前端

运行方式：
    streamlit run app.py
"""

import streamlit as st
import tempfile
import os
import pandas as pd
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# ── 页面配置 ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="医疗多模态智能咨询系统",
    page_icon="🏥",
    layout="wide",
)

# ── 侧边栏 ────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏥 系统说明")
    st.markdown("""
    本系统集成了四个核心技术模块：

    **📷 多模态感知**
    BLIP 图像理解模型，将症状图片转化为文字描述

    **🔍 语义检索 (RAG)**
    FAISS 向量索引 + BGE-small-zh 嵌入，从 385 种疾病知识库中检索相关内容

    **📊 贝叶斯推断**
    朴素贝叶斯 + Laplace 平滑，计算每种疾病的后验概率

    **🤖 多智能体协作**
    CrewAI 驱动三个专科 Agent 分工协作：
    - 症状分析 → 鉴别诊断 → 报告生成

    ---
    > ⚠️ 本系统仅供学习研究使用，**不作为医疗诊断依据**。
    """)

    st.divider()
    st.caption("技术栈：DeepSeek · CrewAI · BLIP · FAISS · Naive Bayes")


# ── 主页面 ────────────────────────────────────────────────────────────
st.title("🏥 医疗多模态智能咨询系统")
st.caption("RAG 检索增强 · 贝叶斯概率推断 · CrewAI 多智能体协作")
st.divider()

col_input, col_image = st.columns([2, 1])

with col_input:
    user_input = st.text_area(
        "请描述您的症状",
        placeholder="例如：皮肤出现红色斑块，伴有瘙痒，已持续三天...",
        height=120,
    )

with col_image:
    image_input = st.file_uploader(
        "上传症状相关图片（可选）",
        type=["png", "jpg", "jpeg"],
    )
    if image_input:
        st.image(image_input, caption="已上传图片", use_container_width=True)

st.divider()

# ── 分析按钮 ──────────────────────────────────────────────────────────
run_button = st.button(
    "🔍 开始分析",
    type="primary",
    disabled=not user_input.strip(),
    use_container_width=True,
)

if run_button:
    image_path = None
    try:
        # 将上传文件写入临时文件，分析后统一清理
        if image_input is not None:
            suffix = os.path.splitext(image_input.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(image_input.getbuffer())
                image_path = tmp.name

        with st.spinner("系统分析中，请稍候..."):
            # 延迟导入避免启动时加载重型模型
            from agents.crew_system import MedicalAgentSystem
            system = MedicalAgentSystem()
            result = system.run(user_input.strip(), image_path)

        # ── 展示中间结果 ──────────────────────────────────────
        col_bayes, col_image_desc = st.columns([1, 1])

        with col_bayes:
            st.subheader("📊 贝叶斯概率推断")
            if result["bayesian"]:
                df = pd.DataFrame(result["bayesian"])
                df.columns = ["候选疾病", "后验概率"]
                df["后验概率"] = df["后验概率"].apply(lambda x: f"{x:.2%}")
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("未能提取有效症状词")

        with col_image_desc:
            if result["image_description"]:
                st.subheader("🖼️ 图像分析结果")
                st.info(result["image_description"])

        with st.expander("📚 RAG 检索到的医学知识库内容", expanded=False):
            st.text(result["retrieved"])

        st.divider()

        # ── 最终报告 ──────────────────────────────────────────
        st.subheader("📋 最终医疗建议报告")
        st.markdown(result["report"])

        st.warning(
            "⚠️ 本报告由 AI 系统生成，仅供参考，不能替代执业医师的专业诊断。"
            "如症状持续或加重，请及时前往正规医疗机构就诊。"
        )

    except ValueError as e:
        st.error(f"配置错误：{e}")
    except Exception as e:
        st.error(f"系统运行出错：{e}")
        raise
    finally:
        # 无论成功或失败，确保临时文件被清理
        if image_path and os.path.exists(image_path):
            os.remove(image_path)
