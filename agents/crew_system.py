# ========= 基础导入 =========
from crewai import Agent, Task, Crew, Process
from crewai.llm import LLM
import os
import sys

# ========= 路径设置 =========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.append(PROJECT_ROOT)

# ========= 引入模块 =========
from probabilistic.bayesian_diagnosis import BayesianDiagnosisEngine
from rag.retriever import retrieve
from multimodal.blip_analyzer import analyze_image

# ========= 初始化贝叶斯引擎 =========
bayes_engine = BayesianDiagnosisEngine(
    os.path.join(PROJECT_ROOT, "data/medical_data.json")
)

# ========= DeepSeek LLM =========
deepseek_llm = LLM(
    model="deepseek-chat",
    provider="openai",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
    temperature=0.3
)

# ========= Agents =========
symptom_agent = Agent(
    role="症状分析专家",
    goal="分析用户症状并提取关键信息",
    backstory="你是一位经验丰富的医生，擅长分析患者症状。",
    llm=deepseek_llm,
    verbose=True
)

diagnosis_agent = Agent(
    role="疾病推理专家",
    goal="根据症状推理可能疾病",
    backstory="你是一名临床医学专家，擅长疾病鉴别诊断。",
    llm=deepseek_llm,
    verbose=True
)

report_agent = Agent(
    role="医疗报告生成专家",
    goal="生成清晰、结构化的医疗报告",
    backstory="你擅长生成专业医学建议和报告。",
    llm=deepseek_llm,
    verbose=True
)

# ========= 主运行函数 =========
def run_medical_system(user_query, image_path=None):

    print("\n==============================")
    print("🚑 医疗多模态智能系统启动")
    print("==============================\n")

    # 1️⃣ 图像分析
    image_description = ""
    if image_path:
        try:
            print("🖼 正在分析图像...")
            image_description = analyze_image(image_path)
            print("🖼 图像分析结果：", image_description)
        except Exception as e:
            print("❌ 图像分析失败：", e)

    # 2️⃣ 融合语义
    enhanced_query = user_query + " " + image_description

    # 3️⃣ RAG 检索
    retrieved_knowledge = retrieve(enhanced_query)

    print("\n📚 检索到的医学知识：")
    print(retrieved_knowledge)

    # 4️⃣ 贝叶斯概率推断（⚠ 注意现在在函数内部）
    bayesian_results = bayes_engine.predict(user_query)

    print("\n📊 贝叶斯概率推断结果：")
    for item in bayesian_results:
        print(f"{item['disease']} 概率: {item['probability']}")

    # 5️⃣ 多 Agent 任务
    task1 = Task(
        description=f"""
用户症状：
{user_query}

图像分析结果：
{image_description}

医学数据库信息：
{retrieved_knowledge}

贝叶斯概率预测结果：
{bayesian_results}

请结合概率结果与医学知识进行综合分析。
""",
        agent=symptom_agent,
        expected_output="结构化症状分析"
    )

    task2 = Task(
        description="根据前面的症状分析，给出可能疾病及原因说明。",
        agent=diagnosis_agent,
        expected_output="疾病列表及分析"
    )

    task3 = Task(
        description="生成最终医疗建议报告，包含免责声明。",
        agent=report_agent,
        expected_output="完整医疗报告"
    )

    # 6️⃣ 创建 Crew
    crew = Crew(
        agents=[symptom_agent, diagnosis_agent, report_agent],
        tasks=[task1, task2, task3],
        process=Process.sequential,
        verbose=True
    )

    # 7️⃣ 执行
    result = crew.kickoff()

    return result


# ========= CLI 入口 =========
if __name__ == "__main__":

    user_input = input("请输入症状：")
    image_input = input("请输入图片路径（没有就回车）：")

    if image_input.strip() == "":
        result = run_medical_system(user_input)
    else:
        result = run_medical_system(user_input, image_input)

    print("\n====== 最终医疗建议 ======\n")
    print(result)