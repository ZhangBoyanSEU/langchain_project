import os
import re
import json
import hashlib
import argparse
from pathlib import Path

import numpy as np

from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
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


def _dedupe_by_id(chunks, ids):
    """按 ID 去重，保留首次出现的 (chunk, id) 对。"""
    seen = {}
    for chunk, chunk_id in zip(chunks, ids):
        if chunk_id not in seen:
            seen[chunk_id] = (chunk, chunk_id)
    deduped_chunks = [pair[0] for pair in seen.values()]
    deduped_ids = [pair[1] for pair in seen.values()]
    if len(deduped_ids) < len(ids):
        print(f"去重: {len(ids)} -> {len(deduped_ids)} 个文本块（跳过 {len(ids) - len(deduped_ids)} 个重复ID）")
    return deduped_chunks, deduped_ids


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

def split_to_chunks(docs):
    """按 split_method 分流切分文档，返回文本块列表。

    - markdown 文件用 MarkdownHeaderTextSplitter 先按标题结构划分，再用 RecursiveCharacterTextSplitter 控制大小
    - 其他文档直接用 RecursiveCharacterTextSplitter 切分
    """
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
    return chunks


def _load_existing_ids_from_jsonl(jsonl_path):
    """读取 JSONL 文件中已有的 id 集合，容错跳过损坏行。"""
    existing_ids = set()
    if not jsonl_path.exists():
        return existing_ids
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                existing_ids.add(record["id"])
            except (json.JSONDecodeError, KeyError):
                print(f"警告: 跳过损坏行: {line[:50]}")
    return existing_ids


def save_chunks_to_jsonl(chunks, jsonl_path):
    """将文本块列表追加写入 JSONL 文件（按 id 去重，同一 id 仅保留一条记录）。"""
    jsonl_path = Path(jsonl_path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    ids = _generate_ids(chunks)
    chunks, ids = _dedupe_by_id(chunks, ids)
    existing_ids = _load_existing_ids_from_jsonl(jsonl_path)

    appended = 0
    skipped = 0
    with jsonl_path.open("a", encoding="utf-8") as f:
        for chunk, chunk_id in zip(chunks, ids):
            if chunk_id in existing_ids:
                skipped += 1
                continue
            record = {
                "id": chunk_id,
                "page_content": chunk.page_content,
                "metadata": chunk.metadata,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            existing_ids.add(chunk_id)
            appended += 1

    total = len(_load_existing_ids_from_jsonl(jsonl_path))
    print(f"已追加 {appended} 个文本块至 {jsonl_path}（跳过重复 {skipped} 条，文件现存 {total} 条）")
    return jsonl_path


def load_chunks_from_jsonl(jsonl_path):
    """从 JSONL 文件加载文本块，返回 (chunks, ids)。"""
    jsonl_path = Path(jsonl_path)
    chunks = []
    ids = []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            chunks.append(
                Document(page_content=record["page_content"], metadata=record["metadata"])
            )
            ids.append(record["id"])

    print(f"从 {jsonl_path} 加载了 {len(chunks)} 个文本块")
    return chunks, ids


def build_vectorstore_from_chunks(chunks, ids, persist_dir="./chroma_db"):
    """将已切分的文本块向量化并存入 Chroma（入库前按 ID 去重）。"""
    chunks, ids = _dedupe_by_id(chunks, ids)
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


def build_vectorstore(docs, persist_dir="./chroma_db"):
    """将文档分块、向量化并存入 Chroma（兼容旧接口）。"""
    chunks = split_to_chunks(docs)
    ids = _generate_ids(chunks)
    return build_vectorstore_from_chunks(chunks, ids, persist_dir)

def process_folder(folder_path, persist_dir="./chroma_db", jsonl_dir="./data/chunks", mode="all"):
    """处理文件夹中的所有文档（md/docx/knowledge）。

    Args:
        folder_path: 包含知识文档的文件夹路径
        persist_dir: 向量库持久化路径
        jsonl_dir: JSONL 文件输出/读取目录
        mode: 处理模式
            "all"       — 分割保存 JSONL + 向量化（默认）
            "split"     — 仅分割并保存为 JSONL，不向量化
            "vectorize" — 仅从 JSONL 加载并向量化，不读取原始文档
    """
    if mode not in ("all", "split", "vectorize"):
        raise ValueError(f"不支持的模式: {mode}，可选: all / split / vectorize")

    supported_exts = (".md", ".markdown", ".docx", ".knowledge")

    if mode in ("all", "split"):
        folder = Path(folder_path)
        if not folder.is_dir():
            raise ValueError(f"路径不是文件夹: {folder_path}")

        files = [f for f in folder.iterdir() if f.suffix.lower() in supported_exts]
        if not files:
            raise ValueError(f"文件夹中未找到支持的文档: {folder_path}")

        print(f"找到 {len(files)} 个文档")
        combined_jsonl_path = Path(jsonl_dir) / "all_chunks.jsonl"
        all_chunks = []
        for file_path in files:
            print(f"正在加载: {file_path.name}")
            docs = load_document(str(file_path))
            chunks = split_to_chunks(docs)
            save_chunks_to_jsonl(chunks, combined_jsonl_path)
            if mode == "all":
                all_chunks.extend(chunks)

        if mode == "split":
            print("仅分割模式完成，跳过向量化。")
            return None

        ids = _generate_ids(all_chunks)
        all_chunks, ids = _dedupe_by_id(all_chunks, ids)
        print(f"共 {len(all_chunks)} 个文本块待向量化")
        return build_vectorstore_from_chunks(all_chunks, ids, persist_dir)

    else:  # mode == "vectorize"
        jsonl_folder = Path(jsonl_dir)
        if not jsonl_folder.is_dir():
            raise FileNotFoundError(
                f"JSONL 目录不存在: {jsonl_dir}，请先以 split 或 all 模式生成"
            )

        jsonl_files = sorted(jsonl_folder.glob("*.jsonl"))
        if not jsonl_files:
            raise ValueError(f"JSONL 目录中未找到 .jsonl 文件: {jsonl_dir}")

        print(f"找到 {len(jsonl_files)} 个 JSONL 文件")
        all_chunks = []
        all_ids = []
        for jsonl_path in jsonl_files:
            print(f"正在加载: {jsonl_path.name}")
            chunks, ids = load_chunks_from_jsonl(jsonl_path)
            all_chunks.extend(chunks)
            all_ids.extend(ids)

        print(f"共 {len(all_chunks)} 个文本块待向量化")
        all_chunks, all_ids = _dedupe_by_id(all_chunks, all_ids)
        return build_vectorstore_from_chunks(all_chunks, all_ids, persist_dir)

def load_vectorstore(persist_dir="./chroma_db"):
    """加载已有的向量库"""
    if not os.path.exists(persist_dir):
        raise FileNotFoundError(f"向量库不存在: {persist_dir}，请先运行 document_processor.py 处理文档")
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=_get_embeddings()
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="处理文档（md/docx/knowledge）：分割为 JSONL 和/或向量化入库")
    parser.add_argument("--folder_path", default="./data/origin", help="包含知识文档的文件夹路径")
    parser.add_argument("--jsonl-dir", default="./data/chunks", help="JSONL 文件输出/读取目录 (默认: ./data/chunks)")
    parser.add_argument("--mode", choices=["all", "split", "vectorize"], default="all",
                        help="处理模式: all=分割+向量化(默认), split=仅分割保存JSONL, vectorize=仅从JSONL向量化")
    parser.add_argument("--persist-dir", default="./chroma_db", help="向量库持久化路径 (默认: ./chroma_db)")
    args = parser.parse_args()

    process_folder(args.folder_path, args.persist_dir, args.jsonl_dir, args.mode)
