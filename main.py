from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# LLM
from langchain_openai import ChatOpenAI          # 使用 OpenAI
from langchain_ollama import ChatOllama    # 使用本地 Ollama

from document_processor import load_vectorstore


def create_qa_chain(vectorstore):
    """创建检索问答链"""
    # 选择 LLM（二选一）

    # 方式一：OpenAI（需设置 OPENAI_API_KEY 环境变量）
    # llm_openai = ChatOpenAI(
    #     model="gpt-3.5-turbo",
    #     temperature=0,
    #     api_key=os.getenv("OPENAI_API_KEY")   # 或直接填写
    # )

    # 方式二：本地 Ollama（需先安装并启动 ollama，且已拉取模型）
    llm_ollama = ChatOllama(model="qwen2.5", temperature=0)

    prompt = ChatPromptTemplate.from_template('''
基于以下上下文回答问题。如果上下文中没有相关信息，请说明你不知道。

上下文:
{context}

问题: {input}

回答:''')

    combine_docs_chain = create_stuff_documents_chain(llm_ollama, prompt)
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
