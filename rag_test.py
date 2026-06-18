import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from langchain_docling import DoclingLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
import requests
import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore


# 系统提示词：定义智能助手的角色和工具使用规则
prompt = """
你是一个智能助手。

你拥有三个工具：

1.get_weather
查询天气

2.get_sum
计算数字

3.rag_search
查询知识库内容

如果用户询问知识库中的内容，
优先调用rag_search。

输出结果同时告诉我调用了什么工具。
"""

# 定义工具描述（告诉 LLM 有哪些工具可用）
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的实时天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_sum",
            "description": "计算两个数的和",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "string", "description": "第一个数"},
                    "y": {"type": "string", "description": "第二个数"}
                },
                "required": ["x", "y"]
            }
        }
    },
{
    "type": "function",
    "function": {
        "name": "rag_search",
        "description": "查询知识库中的内容",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "用户的问题"
                }
            },
            "required": ["question"]
        }
    }
}
]

# 加载PDF文档并构建向量数据库
FILE_PATH = "your file path"
loader = DoclingLoader(file_path=FILE_PATH)
document = loader.load()

# for document in loader.lazy_load():

# Text splitter integrations
text_splitting = RecursiveCharacterTextSplitter(chunk_size = 500,chunk_overlap = 50)
chunks = text_splitting.split_documents(document)

embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/e5-large-v2",
    encode_kwargs={"prompt": "passage: "},
    query_encode_kwargs={"prompt": "query: "},
)

vector_store = InMemoryVectorStore(embedding=embeddings)
vector_store.add_documents(documents=chunks)


def get_sum(x : str,y : str) ->str:
    """
    计算两个数字字符串的和

    Args:
        x (str): 第一个数字的字符串表示
        y (str): 第二个数字的字符串表示

    Returns:
        str: 两个数字相加后的结果字符串
    """
    return str(int(x) + int(y))


def get_weather(city : str) ->str:
    """
    查询指定城市的实时天气信息
    Args:
        city (str): 要查询天气的城市名称
    Returns:
        str: 包含城市、天气描述和温度的格式化字符串
    """
    url = f"https://wttr.in/{city}?format=j1"
    response = requests.get(url)

    # 解析返回的JSON数据
    data = response.json()
    current_condition = data["current_condition"][0]
    weather_desc = current_condition['weatherDesc'][0]['value']
    temp_c = current_condition['temp_C']
    return f"{city}当前天气:{weather_desc}，气温{temp_c}摄氏度"

    # print(weather_desc,temp_c)


def rag_search(question: str) -> str:
    """
    基于RAG（检索增强生成）的知识库搜索功能

    从向量数据库中检索与问题最相似的文档片段，
    并基于这些上下文生成回答

    Args:
        question (str): 用户提出的问题

    Returns:
        str: 基于知识库上下文的回答，如果未找到相关内容则返回提示信息
    """
    results = vector_store.similarity_search(
        question,
        k=3
    )

    context = "\n".join(
        [doc.page_content for doc in results]
    )

    rag_prompt = f"""
你是知识库助手。

请严格依据上下文回答问题。

如果上下文没有答案，
请回答：

知识库中未找到相关内容

上下文：

{context}

问题：

{question}
"""

    completion = client.chat.completions.create(
        model="qwen3.6-plus",
        messages=[
            {
                "role":"user",
                "content":rag_prompt
            }
        ]
    )

    return completion.choices[0].message.content


# 初始化 OpenAI 客户端，配置阿里云 DashScope API
client = OpenAI(

    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 获取用户输入并进行流式对话处理
user_input = input("输入你的问题：")

# 创建流式聊天补全请求
completion = client.chat.completions.create(

    model="qwen3.6-flash-2026-04-16",
    messages=[
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_input},

    ],
    tools=tools,
    tool_choice="auto",
    stream=False
)


# --- 处理 tool calls ---
msg = completion.choices[0].message
messages = [{"role": "system", "content": prompt}, {"role": "user", "content": user_input}]

# 可用函数映射
available_functions = {
    "get_weather": get_weather,
    "get_sum": get_sum,
    "rag_search": rag_search,
}

# 循环处理模型的工具调用请求，直到不需要调用工具为止
while msg.tool_calls:
    # 把模型的 tool_call 消息加入对话
    messages.append(msg)

    for tool_call in msg.tool_calls:
        func_name = tool_call.function.name
        func_args = json.loads(tool_call.function.arguments)

        print(f"🔧 调用工具: {func_name}({func_args})")

        # 执行工具函数
        func = available_functions.get(func_name)
        result = func(**func_args) if func else f"未知工具: {func_name}"

        # 把工具结果加入对话
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": str(result)
        })

    # 再次请求模型，获取最终回复
    completion = client.chat.completions.create(
        model="qwen3.6-plus",
        messages=messages,
        tools=tools,
        stream=False
    )
    msg = completion.choices[0].message

# 输出最终文本
print(msg.content)

