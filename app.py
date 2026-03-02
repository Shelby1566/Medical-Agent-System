import streamlit as st
from agents.crew_system import run_medical_system

st.title("医疗多模态智能咨询系统")

# 输入症状
user_input = st.text_input("请输入症状:")

# 上传图片（可选）
image_input = st.file_uploader("上传症状相关图片（可选）", type=["png", "jpg", "jpeg"])

if st.button("生成报告"):
    image_path = None
    if image_input is not None:
        image_path = f"temp_{image_input.name}"
        with open(image_path, "wb") as f:
            f.write(image_input.getbuffer())
    # 调用你的 Crew 系统
    result = run_medical_system(user_input, image_path)
    st.subheader("最终医疗建议")
    st.text(result)
