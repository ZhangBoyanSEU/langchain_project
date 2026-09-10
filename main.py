from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# LLM
from langchain_openai import ChatOpenAI    # 使用 OpenAI
from langchain_ollama import ChatOllama    # 使用本地 Ollama

from data_processor import load_vectorstore



def create_map_reduce_documents_chain(llm, map_prompt, reduce_prompt):
    """创建 Map-Reduce 文档组合链。

    与 create_stuff_documents_chain 接口一致：
      输入: {"context": [Document, ...], "input": str}
      输出: {"answer": str, "context": [Document, ...]}

    - Map 阶段：对每个文档独立调用 LLM 生成子回答
    - Reduce 阶段：将所有子回答拼接后调用 LLM 综合出最终回答
    """

    map_chain = map_prompt | llm | StrOutputParser()
    reduce_chain = reduce_prompt | llm | StrOutputParser()

    def _map_reduce(inputs):
        docs = inputs["context"]
        question = inputs["input"]

        # --- Map：逐文档独立回答 ---
        sub_answers = []
        for doc in docs:
            result = map_chain.invoke({"context": doc.page_content, "input": question})
            sub_answers.append(result)

        # --- Reduce：综合所有子回答 ---
        combined = "\n\n".join(sub_answers)
        final_answer = reduce_chain.invoke({"context": combined, "input": question})

        return {"answer": final_answer, "context": docs}

    return RunnableLambda(_map_reduce)


def create_qa_chain(vectorstore, strategy="stuff"):
    """创建检索问答链

    Args:
        vectorstore: 向量库实例
        strategy: 文档组合策略
            "stuff"       — 全量塞入
            "map_reduce"  — Map-Reduce 分批处理
    """
    # 选择 LLM（二选一）

    # 方式一：OpenAI（需设置 OPENAI_API_KEY 环境变量）
    # llm_openai = ChatOpenAI(
    #     model="gpt-3.5-turbo",
    #     temperature=0,
    #     api_key=os.getenv("OPENAI_API_KEY")   # 或直接填写
    # )

    # 方式二：本地 Ollama（需先安装并启动 ollama，且已拉取模型）
    llm_ollama = ChatOllama(model="llama3.2", temperature=0.2)

    # Stuff 策略的 Prompt
    stuff_prompt = ChatPromptTemplate.from_template('''
基于以下上下文回答问题。如果上下文中没有相关信息，请说明你不知道。

上下文:
{context}

问题: {input}

回答:''')

    if strategy == "map_reduce":
        map_prompt = ChatPromptTemplate.from_template('''
基于以下文档片段回答问题。如果片段中没有相关信息，请返回"未知"。

文档片段:
{context}

问题: {input}

回答:''')

        reduce_prompt = ChatPromptTemplate.from_template('''
以下是基于多个文档片段的独立回答，请综合它们给出最终回答。如果所有回答都是"未知"，请说明你不知道。

独立回答:
{context}

问题: {input}

最终回答:''')

        combine_docs_chain = create_map_reduce_documents_chain(
            llm_ollama, map_prompt, reduce_prompt
        )
    else:
        combine_docs_chain = create_stuff_documents_chain(llm_ollama, stuff_prompt)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return create_retrieval_chain(retriever, combine_docs_chain)


def main():
    # 1. 加载数据库
    print("正在加载数据库...")
    vectorstore = load_vectorstore()
    print("数据库加载完毕")

    # 2. 创建问答链
    print("正在创建问答链...")
    qa_chain = create_qa_chain(vectorstore)
    print("问答链创建完毕")

    # 3. 交互式问答
    print("\n知识库就绪！输入问题开始提问（输入 exit 退出）")
    while True:
        query = input("\n问题: ").strip()
        if query.lower() in ("exit", "quit", "q"):
            break
        if not query:
            continue

        result = qa_chain.invoke({"input": query})
        print("\n回答:", result["answer"])
        print("\n参考来源:")
        for i, doc in enumerate(result["context"], 1):
            print(f"{i}. {doc.page_content[:100]}...")


if __name__ == "__main__":
    main()
