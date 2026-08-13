import os
import hashlib
import argparse
from pathlib import Path

# 文档加载器
from langchain_unstructured import UnstructuredLoader

# 文本分割
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 嵌入模型与向量库
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def _get_embeddings():
    # os.environ.setdefault("HF_HUB_OFFLINE", "1")
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        encode_kwargs={"normalize_embeddings": True}
    )


def _generate_ids(chunks):
    """根据文本内容生成确定性ID，相同内容产生相同ID，实现去重(upsert)"""
    return [
        hashlib.sha256(chunk.page_content.encode("utf-8")).hexdigest()
        for chunk in chunks
    ]


def load_document(file_path):
    ext = Path(file_path).suffix.lower()
    if ext in (".knowledge", ".doc", ".docx"):
        loader = UnstructuredLoader(file_path=file_path)
    else:
        raise ValueError("仅支持 .doc/.knowledge/.docx 文件")
    return loader.load()

def build_vectorstore(docs, persist_dir="./chroma_db"):
    """将文档分块、向量化并存入 Chroma"""
    # 1. 分割文本
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    unique_count = len(set(c.page_content for c in chunks))
    print(f"共切分为 {len(chunks)} 个文本块（其中 {unique_count} 个唯一内容）")

    # 2. 生成确定性ID（基于内容哈希），相同内容ID相同，实现upsert去重
    ids = _generate_ids(chunks)

    # 3. 创建/加载向量库
    if os.path.exists(persist_dir):
        print("检测到已有向量库，追加/更新文档...")
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=_get_embeddings()
        )
        vectorstore.delete(ids=ids)
        vectorstore.add_documents(documents=chunks, ids=ids)
    else:
        print("创建新的向量库...")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=_get_embeddings(),
            ids=ids,
            persist_directory=persist_dir
        )
    return vectorstore

def process_folder(folder_path, persist_dir="./chroma_db"):
    """处理文件夹中的所有Word文档并存入向量库"""
    folder = Path(folder_path)
    if not folder.is_dir():
        raise ValueError(f"路径不是文件夹: {folder_path}")

    word_files = [f for f in folder.iterdir() if f.suffix.lower() in (".doc", ".docx")]
    if not word_files:
        raise ValueError(f"文件夹中未找到Word文档: {folder_path}")

    print(f"找到 {len(word_files)} 个Word文档")
    all_docs = []
    for file_path in word_files:
        print(f"正在加载: {file_path.name}")
        docs = load_document(str(file_path))
        all_docs.extend(docs)

    print(f"共加载 {len(all_docs)} 个文档片段")
    return build_vectorstore(all_docs, persist_dir)

def load_vectorstore(persist_dir="./chroma_db"):
    """加载已有的向量库"""
    if not os.path.exists(persist_dir):
        raise FileNotFoundError(f"向量库不存在: {persist_dir}，请先运行 document_processor.py 处理文档")
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=_get_embeddings()
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="处理Word文档并存入向量数据库")
    parser.add_argument("folder_path", help="包含Word文档的文件夹路径")
    parser.add_argument("--persist-dir", default="./chroma_db", help="向量库持久化路径 (默认: ./chroma_db)")
    args = parser.parse_args()

    process_folder(args.folder_path, args.persist_dir)
