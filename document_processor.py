import os
import re
import hashlib
import argparse
from pathlib import Path

import numpy as np

# 文档加载器
from langchain_unstructured import UnstructuredLoader

# 文本分割
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

# 文档模型与嵌入模型与向量库
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def _get_embeddings():
    # os.environ.setdefault("HF_HUB_OFFLINE", "1")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )

def _generate_ids(chunks):
    """根据文本内容生成确定性ID，相同内容产生相同ID，实现去重(upsert)"""
    return [
        hashlib.sha256(chunk.page_content.encode("utf-8")).hexdigest()
        for chunk in chunks
    ]


class SemanticChunker:
    """语义分块器：基于相邻句子的嵌入相似度，在语义"断崖"处自动切分。

    算法流程：
      1. 按中英文标点将文本拆分为句子
      2. 对每个句子计算嵌入向量（需传入已开启 normalize_embeddings 的模型）
      3. 计算相邻句子的余弦相似度（normalize 后即点积）
      4. 根据阈值策略确定断点：相似度低于阈值处切分
      5. 将断点之间的句子合并为 chunk

    阈值策略（breakpoint_threshold_type）：
      - "percentile"：取相似度分布的第 N 百分位（breakpoint_threshold_amount，默认 95）
      - "standard_deviation"：均值 - N × 标准差
      - "interquartile"：第三四分位 - 1.5 × IQR
    """

    def __init__(
        self,
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,
        chunk_size=500,
    ):
        self._embeddings = embeddings
        self._breakpoint_threshold_type = breakpoint_threshold_type
        self._breakpoint_threshold_amount = breakpoint_threshold_amount
        self._chunk_size = chunk_size

    @staticmethod
    def _split_sentences(text):
        """按中英文句末标点和换行符将文本拆分为句子列表。"""
        sentences = re.split(r'(?<=[。！？.!?])\s*|\n+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _compute_similarities(self, sentences):
        """计算相邻句子的余弦相似度序列。"""
        if len(sentences) < 2:
            return []
        embeddings = self._embeddings.embed_documents(sentences)
        vectors = np.array(embeddings)
        similarities = [
            float(np.dot(vectors[i], vectors[i + 1]))
            for i in range(len(vectors) - 1)
        ]
        return similarities

    def _find_breakpoints(self, similarities):
        """根据阈值策略返回断点索引列表。"""
        if not similarities:
            return []

        similarities_arr = np.array(similarities)

        if self._breakpoint_threshold_type == "percentile":
            threshold = float(np.percentile(similarities_arr, self._breakpoint_threshold_amount))
        elif self._breakpoint_threshold_type == "standard_deviation":
            threshold = float(
                np.mean(similarities_arr)
                - self._breakpoint_threshold_amount * np.std(similarities_arr)
            )
        elif self._breakpoint_threshold_type == "interquartile":
            q1, q3 = np.percentile(similarities_arr, [25, 75])
            threshold = float(q3 - 1.5 * (q3 - q1))
        else:
            raise ValueError(
                f"不支持的阈值类型: {self._breakpoint_threshold_type}，"
                f"可选: percentile / standard_deviation / interquartile"
            )

        return [i for i, sim in enumerate(similarities) if sim < threshold]

    def split_text(self, text):
        """将文本按语义断点切分为 chunk 列表。"""
        sentences = self._split_sentences(text)
        if not sentences:
            return []

        if len(sentences) == 1:
            return [sentences[0]]

        similarities = self._compute_similarities(sentences)
        breakpoints = self._find_breakpoints(similarities)

        chunks = []
        start = 0
        for bp in breakpoints:
            bp += 1  # 断点在 i 与 i+1 之间，切分点取 i+1
            chunk = " ".join(sentences[start:bp])
            if chunk:
                chunks.append(chunk)
            start = bp
        if start < len(sentences):
            chunk = " ".join(sentences[start:])
            if chunk:
                chunks.append(chunk)

        return chunks

    def split_documents(self, documents):
        """对 Document 列表逐个进行语义分块，保留原始 metadata。"""
        result = []
        for doc in documents:
            chunks = self.split_text(doc.page_content)
            for chunk in chunks:
                result.append(Document(page_content=chunk, metadata=dict(doc.metadata)))
        return result


def load_document(file_path):
    ext = Path(file_path).suffix.lower()
    if ext in (".md", ".markdown"):
        text = Path(file_path).read_text(encoding="utf-8")
        return [Document(page_content=text, metadata={"source": file_path, "split_method": "markdown"})]
    elif ext in (".knowledge", ".docx"):
        loader = UnstructuredLoader(file_path=file_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata["split_method"] = "recursive"
        return docs
    else:
        raise ValueError("仅支持 .md/.markdown/.docx/.knowledge 文件")

def build_vectorstore(docs, persist_dir="./chroma_db"):
    """将文档分块、向量化并存入 Chroma"""
    # 1. 按 split_method 分流：markdown 文件用 Markdown 结构划分，其他用递归字符划分
    md_docs = [d for d in docs if d.metadata.get("split_method") == "markdown"]
    non_md_docs = [d for d in docs if d.metadata.get("split_method") != "markdown"]
    chunks = []

    if md_docs:
        md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "H1"),
                ("##", "H2"),
                ("###", "H3"),
            ],
            strip_headers=False,
        )
        size_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""],
        )
        for doc in md_docs:
            md_chunks = md_splitter.split_text(doc.page_content)
            for mc in md_chunks:
                merged_meta = {**doc.metadata, **mc.metadata}
                sized = size_splitter.split_documents(
                    [Document(page_content=mc.page_content, metadata=merged_meta)]
                )
                chunks.extend(sized)

    if non_md_docs:
        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""],
        )
        chunks.extend(recursive_splitter.split_documents(non_md_docs))

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
    """处理文件夹中的所有文档（md/docx/knowledge）并存入向量库"""
    folder = Path(folder_path)
    if not folder.is_dir():
        raise ValueError(f"路径不是文件夹: {folder_path}")

    supported_exts = (".md", ".markdown", ".docx", ".knowledge")
    files = [f for f in folder.iterdir() if f.suffix.lower() in supported_exts]
    if not files:
        raise ValueError(f"文件夹中未找到支持的文档: {folder_path}")

    print(f"找到 {len(files)} 个文档")
    all_docs = []
    for file_path in files:
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
    parser = argparse.ArgumentParser(description="处理文档（md/docx/knowledge）并存入向量数据库")
    parser.add_argument("--folder_path", default="./data", help="包含知识文档的文件夹路径")
    parser.add_argument("--persist-dir", default="./chroma_db", help="向量库持久化路径 (默认: ./chroma_db)")
    args = parser.parse_args()

    process_folder(args.folder_path, args.persist_dir)
