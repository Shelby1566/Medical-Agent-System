"""
医疗多智能体协作系统

三个 Agent 各司其职、真正分工协作：

  症状分析 Agent  →  结构化症状分类 + 严重程度评估
       ↓ context
  鉴别诊断 Agent  →  基于临床推理，逐一验证/挑战贝叶斯结果
       ↓ context
  报告生成 Agent  →  综合两位专家意见，生成患者友好的结构化报告

核心设计：
- 鉴别诊断 Agent 显式对贝叶斯概率模型的结果进行验证和质疑，
  体现"统计模型 + LLM 临床推理"的双重验证机制。
- MedicalAgentSystem 类封装所有状态，避免全局变量，支持并发。
- 所有重型模型（BLIP、SentenceTransformer）在对应模块中懒加载。
"""

from crewai import Agent, Task, Crew, Process
from crewai.llm import LLM
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from probabilistic.bayesian_diagnosis import BayesianDiagnosisEngine
from rag.retriever import retrieve
from multimodal.blip_analyzer import analyze_image

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "medical_data.json")


class MedicalAgentSystem:
    """
    医疗多智能体系统主类。

    使用方式：
        system = MedicalAgentSystem()
        result = system.run("持续咳嗽 低烧 盗汗")
        print(result["report"])
    """

    def __init__(self):
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("环境变量 DEEPSEEK_API_KEY 未设置，请在 .env 文件中配置。")

        self._llm = LLM(
            model="deepseek-chat",
            provider="openai",
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
            temperature=0.3,
        )
        self._bayes_engine = BayesianDiagnosisEngine(DATA_PATH)
        self._init_agents()

    def _init_agents(self):
        """初始化三个专科 Agent，角色设定体现分工差异。"""

        self.symptom_agent = Agent(
            role="症状分析专家",
            goal="对用户描述的症状进行系统性结构化分析，提取对鉴别诊断最有价值的信息",
            backstory=(
                "你是一位有20年临床经验的全科主任医师，擅长从患者描述中精准提取症状要素。"
                "你的工作是对症状进行医学规范化整理，为下游鉴别诊断提供高质量的结构化输入。"
                "你注重每个症状的鉴别意义，善于识别哪些症状是'关键区分点'。"
            ),
            llm=self._llm,
            verbose=True,
        )

        self.diagnosis_agent = Agent(
            role="临床鉴别诊断专家",
            goal="运用临床推理，对统计模型给出的候选诊断进行逐一验证，必要时提出挑战或补充",
            backstory=(
                "你是一位内科主任医师，专注于疑难病例的鉴别诊断。"
                "你的核心能力是：不盲目接受统计概率结果，而是基于症状特征和医学知识进行独立的临床推理，"
                "能够指出概率模型的遗漏或误判，给出有充分依据的鉴别诊断排序。"
            ),
            llm=self._llm,
            verbose=True,
        )

        self.report_agent = Agent(
            role="医疗报告撰写专家",
            goal="将专家诊断意见整合为患者可理解的结构化报告，兼顾专业性与可读性",
            backstory=(
                "你是一位善于医患沟通的主治医师，擅长将复杂的临床推理转化为通俗易懂的医疗建议。"
                "你始终将患者安全放在首位，会明确指出何时需要立即就医，"
                "并在每份报告中附上规范的 AI 免责声明。"
            ),
            llm=self._llm,
            verbose=True,
        )

    def run(self, user_query: str, image_path: str = None) -> dict:
        """
        运行完整的多模态医疗分析流程。

        Args:
            user_query:  用户描述的症状文本
            image_path:  可选，症状相关图片路径

        Returns:
            {
                "bayesian":          贝叶斯 Top-3 预测结果 (list),
                "retrieved":         RAG 检索到的医学知识 (str),
                "image_description": BLIP 图像描述 (str),
                "report":            最终医疗报告 (str),
            }
        """
        print("\n" + "=" * 50)
        print("  医疗多模态智能系统启动")
        print("=" * 50)

        # ── Step 1：多模态图像分析 ────────────────────────────
        image_description = ""
        if image_path:
            try:
                print("\n[1/4] 图像分析中...")
                image_description = analyze_image(image_path)
                print(f"      图像描述：{image_description}")
            except Exception as e:
                print(f"      图像分析失败（已跳过）：{e}")

        # ── Step 2：查询增强 ──────────────────────────────────
        enhanced_query = f"{user_query} {image_description}".strip()

        # ── Step 3：RAG 语义检索 ──────────────────────────────
        print("\n[2/4] RAG 检索医学知识库...")
        retrieved_knowledge = retrieve(enhanced_query, top_k=3)

        # ── Step 4：贝叶斯概率推断 ────────────────────────────
        print("\n[3/4] 贝叶斯概率推断...")
        bayesian_results = self._bayes_engine.predict(user_query, top_k=3)
        bayesian_text = "\n".join(
            f"  {i+1}. {r['disease']}（后验概率：{r['probability']:.2%}）"
            for i, r in enumerate(bayesian_results)
        )
        print(f"      预测结果：\n{bayesian_text}")

        # ── Step 5：多 Agent 协作 ─────────────────────────────
        print("\n[4/4] 多智能体协作分析...\n")

        image_note = image_description if image_description else "（本次无图像输入）"

        task1 = Task(
            description=f"""你是症状分析专家，请对以下输入进行系统性分析。

【用户描述的症状】
{user_query}

【图像分析结果】
{image_note}

【贝叶斯模型初步预测】（统计概率，仅供参考）
{bayesian_text}

请完成以下分析：
1. 列出所有症状，按身体系统分类（皮肤/呼吸/消化/心血管/神经系统等）
2. 评估症状严重程度（轻度 / 中度 / 重度）及可能的病程阶段
3. 标注 2-3 个"关键鉴别症状"（对区分不同诊断最有价值的症状）
4. 若有图像信息，说明图像特征与症状描述的关联性
""",
            agent=self.symptom_agent,
            expected_output="结构化症状分析，包含：系统分类、严重程度、关键鉴别症状（及图像关联说明）",
        )

        task2 = Task(
            description=f"""你是临床鉴别诊断专家。请在症状分析专家工作的基础上，运用临床推理进行深度鉴别诊断。

【RAG 检索的医学知识库内容】
{retrieved_knowledge}

【贝叶斯统计模型给出的候选疾病】
{bayesian_text}

请完成以下鉴别诊断：
1. 针对贝叶斯模型给出的每个候选疾病，逐一分析：
   - ✅ 支持该诊断的症状依据（与患者症状吻合的方面）
   - ❌ 不支持或需要排除的因素（与患者症状矛盾的方面）
   - 综合评估该疾病的临床可能性

2. 结合医学知识库，判断贝叶斯模型是否遗漏了重要诊断，若有请补充并说明理由

3. 给出你的最终鉴别诊断排序（从最可能到最不可能），每条附上核心推理依据

4. 建议进一步确诊所需的检查项目（血液检查、影像、皮肤活检等）

注意：请明确说明你的临床推理过程，不能仅凭概率数字做决策。
""",
            agent=self.diagnosis_agent,
            context=[task1],
            expected_output="鉴别诊断报告：每个候选疾病的支持/排除分析、最终诊断排序（含推理）、建议检查项目",
        )

        task3 = Task(
            description="""你是医疗报告撰写专家。请综合症状分析专家和鉴别诊断专家的意见，生成一份面向患者的完整医疗建议报告。

报告必须严格按以下结构输出：

## 一、症状总结
（用通俗语言描述患者的主要症状）

## 二、可能诊断（按可能性排序）
（列出 Top-3 诊断，每条说明可能性理由）

## 三、建议检查项目
（具体的检查项目，说明每项检查的目的）

## 四、就医建议
（何时需要立即就医 / 可以先观察 / 建议挂哪个科室）

## 五、生活注意事项
（日常护理、饮食、休息等具体建议）

## ⚠️ 免责声明
本报告由 AI 系统生成，仅供参考，不能替代执业医师的专业诊断。如症状持续或加重，请及时前往正规医疗机构就诊。

要求：语言通俗易懂，建议具体可操作，免责声明必须完整保留。
""",
            agent=self.report_agent,
            context=[task1, task2],
            expected_output="完整的结构化医疗建议报告，包含五个章节和免责声明",
        )

        crew = Crew(
            agents=[self.symptom_agent, self.diagnosis_agent, self.report_agent],
            tasks=[task1, task2, task3],
            process=Process.sequential,
            verbose=True,
        )

        crew_result = crew.kickoff()

        return {
            "bayesian": bayesian_results,
            "retrieved": retrieved_knowledge,
            "image_description": image_description,
            "report": str(crew_result),
        }


# ── 兼容旧接口 ────────────────────────────────────────────────────────
def run_medical_system(user_query: str, image_path: str = None) -> str:
    """向后兼容的函数接口，返回最终报告文本。"""
    system = MedicalAgentSystem()
    return system.run(user_query, image_path)["report"]


# ── CLI 入口 ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

    symptoms = input("请输入症状：")
    img = input("请输入图片路径（没有就回车）：").strip() or None
    result = run_medical_system(symptoms, img)
    print("\n====== 最终医疗建议 ======\n")
    print(result)
