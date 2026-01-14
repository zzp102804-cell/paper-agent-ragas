import os
import tempfile
import math
import streamlit as st
from datasets import Dataset
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.tools.retriever import create_retriever_tool
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.prebuilt import create_react_agent
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.run_config import RunConfig
# ================= 1. 页面配置 =================
st.set_page_config(page_title="全能文献 Agent", layout="wide", page_icon="🎓")
# ================= 2. 样式优化 (CSS) =================
st.markdown("""
<style>
    /* 文件上传区的文字提示 */
    [data-testid="stFileUploaderDropzoneInstructions"] > div > span {visibility: hidden; height: 0;}
    [data-testid="stFileUploaderDropzoneInstructions"] > div > small {visibility: hidden; height: 0;}
    [data-testid="stFileUploaderDropzoneInstructions"] > div::before {
        content: "请将文件拖拽至此";
        visibility: visible;
        display: block;
        font-size: 1rem;
        margin-bottom: 0.5rem;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] > div::after {
        content: "单个文件限制 200MB • PDF, DOCX, TXT";
        visibility: visible;
        display: block;
        font-size: 0.8rem;
        color: #808495;
    }
    [data-testid="stFileUploader"] button[data-testid="baseButton-secondary"] {
        font-size: 0 !important;
    }
    [data-testid="stFileUploader"] button[data-testid="baseButton-secondary"]::after {
        content: "浏览文件";
        font-size: 1rem !important;
        visibility: visible;
    }
</style>
""", unsafe_allow_html=True)
st.title("🎓 全能文献 Agent")
# ================= 3. 侧边栏 (控制台) =================
with st.sidebar:
    st.header("⚙️ 控制台")
    api_key = st.text_input("请输入 OpenRouter/OpenAI API Key", type="password")

    uploaded_files = st.file_uploader(
        "上传论文 (支持多个文件)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )
    st.info("💡 模型已锁定为: Chatgpt-5.2")
    if uploaded_files:
        st.write("📚 **已加载文档列表:**")
        for f in uploaded_files:
            st.caption(f"- {f.name}")

    st.divider()
# ================= 4. 核心工具定义 =================
def save_uploaded_files(uploaded_files):
    file_paths = []
    for uploaded_file in uploaded_files:
        file_extension = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            file_paths.append(tmp_file.name)
    return file_paths

@st.cache_resource
def create_rag_tool(file_paths, api_key):
    """创建检索工具（包含优化后的参数：Overlap, Top-K, Flashrank）"""
    all_docs = []
    for file_path in file_paths:
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            loader = TextLoader(file_path)
        all_docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
    splits = splitter.split_documents(all_docs)

    embeddings = OpenAIEmbeddings(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )

    vectorstore = Chroma.from_documents(splits, embeddings)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 30})

    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 30

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )

    compressor = FlashrankRerank(top_n=10)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever
    )

    tool = create_retriever_tool(
        compression_retriever,
        "search_paper_content",
        "必须优先使用此工具来搜索和查询用户上传的论文内容。包含具体数据、模型、结论等。"
    )
    return tool

@tool
def calculator(expression: str) -> str:
    """计算器工具"""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算出错: {e}"

search_tool = DuckDuckGoSearchRun()
# ================= 5. Agent 初始化 =================
def initialize_agent(rag_tool, api_key):
    llm = ChatOpenAI(
        model="openai/gpt-5.2",
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0
    )
    tools = [rag_tool, calculator, search_tool]
    agent = create_react_agent(llm, tools)
    return agent
# ================= 6. Ragas 评测模块 =================
def run_real_ragas_evaluation(question, answer, contexts, api_key):
    """配置：n=1, temperature=0, timeout=1200s"""
    try:
        data = {
            'question': [question],
            'answer': [answer],
            'contexts': [contexts],
        }
        dataset = Dataset.from_dict(data)

        eval_llm = ChatOpenAI(
            model="openai/gpt-5.2",
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0,
            timeout=1200,
            max_retries=3
        )

        eval_embeddings = OpenAIEmbeddings(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )

        my_run_config = RunConfig(
            timeout=1200,
            max_workers=1,
            max_retries=3
        )

        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=eval_llm,
            embeddings=eval_embeddings,
            raise_exceptions=False,
            run_config=my_run_config
        )
        return result
    except Exception as e:
        print(f"Ragas 内部报错: {str(e)}")
        return {"error": str(e)}
# ================= 7. 主逻辑 =================
if uploaded_files and api_key:
    file_paths = save_uploaded_files(uploaded_files)
    rag_tool = create_rag_tool(file_paths, api_key)

    # Agent 初始化
    if "agent_engine" not in st.session_state:
        st.session_state.agent_engine = initialize_agent(rag_tool, api_key)
        st.toast("Multi-Agent 系统已激活！", icon="🚀")

    # 初始化 Session State
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 显示历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 处理用户输入
    if prompt := st.chat_input("试着问我：给我生成这些论文的文献综述..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            retrieved_contexts = []

            level4_system_prompt = """
            你是一个由 'Researcher' (研究员) 和 'Writer' (作家) 组成的 Level 4 学术智能体。

            1. **Researcher 阶段**: 
               - 当用户提问时，你必须先调用工具 (search_paper_content, DuckDuckGo) 获取事实。
               - 严禁凭空捏造数据。如果没有查到，就说没查到。

            2. **Writer 阶段**:
               - 拿到数据后，以严谨的学术风格（Academic Tone）撰写回答。
               - 引用数据时要具体。
               - 如果涉及多期/多论文数据的对比（如第一期/篇vs第二期/篇vs第三期/篇...），**必须使用 Markdown 表格**进行展示，以便于我很直观地进行对比分析。

            请严格遵循：先思考 -> 决定调用哪个工具 -> 获取结果 -> 最终写作 的流程。
            """

            with st.status("🕵️‍♂️ Agent (Researcher & Writer) 正在协同...", expanded=True) as status:
                messages_input = [
                    SystemMessage(content=level4_system_prompt),
                    HumanMessage(content=prompt)
                ]

                event_stream = st.session_state.agent_engine.stream(
                    {"messages": messages_input},
                    stream_mode="values"
                )

                final_answer = ""

                # 流式输出处理
                for event in event_stream:
                    if "messages" in event:
                        last_msg = event["messages"][-1]
                        if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                            for tool_call in last_msg.tool_calls:
                                status.write(f"🔨 **Researcher**: 调用工具 `{tool_call['name']}`")
                        elif last_msg.type == 'tool':
                            content = str(last_msg.content)
                            preview = content[:50] + "..."
                            status.write(f"📊 **Data Acquired**: {preview}")
                            retrieved_contexts.append(content)
                        elif last_msg.content:
                            if not (hasattr(last_msg, 'tool_calls') and last_msg.tool_calls):
                                final_answer = last_msg.content

                status.update(label="✅ Writer 写作完成", state="complete", expanded=False)

            message_placeholder.markdown(final_answer)
            st.session_state.messages.append({"role": "assistant", "content": final_answer})

            # Ragas 评测
            if retrieved_contexts:
                with st.expander("AI 生成内容 Ragas 评测"):
                    st.info("正在调用 Ragas 库进行相关指标计算 (Faithfulness & Relevancy)...")
                    ragas_result = run_real_ragas_evaluation(prompt, final_answer, retrieved_contexts, api_key)

                    if isinstance(ragas_result, dict) and "error" in ragas_result:
                        st.error(f"Ragas 评测出错: {ragas_result['error']}")
                    else:
                        df_res = ragas_result.to_pandas()
                        f_val = df_res.iloc[0]['faithfulness']
                        r_val = df_res.iloc[0]['answer_relevancy']

                        def format_score(val):
                            if isinstance(val, float) and math.isnan(val):
                                return None
                            return val * 10

                        f_score = format_score(f_val)
                        r_score = format_score(r_val)

                        c1, c2 = st.columns(2)
                        with c1:
                            if f_score is None:
                                st.warning("信度: 评分失败 (API响应异常)")
                            else:
                                st.metric("信度 (Faithfulness)", f"{f_score:.2f}/10",
                                          help="检测是否存在幻觉，是否忠于原文")
                        with c2:
                            if r_score is None:
                                st.warning("相关度: 评分失败 (API响应异常)")
                            else:
                                st.metric("相关度 (Relevance)", f"{r_score:.2f}/10", help="回答是否切题")

else:
    st.info("👈 请在左侧上传论文（支持多篇）并输入 API Key")