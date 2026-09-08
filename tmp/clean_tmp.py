import re
import argparse
import unicodedata
from pathlib import Path
from copy import deepcopy

from docx import Document
from docx.oxml.ns import qn


class DocxCleaner:
    """docx 文件数据清洗器。

    清洗项目：
      1. Unicode NFC 归一化
      2. 空白字符归一化（制表符、不间断空格、连续空格、首尾空白）
      3. 不可见控制字符移除（零宽字符、BOM 等）
      4. 超链接解包（保留显示文本，移除链接标记；同时处理 w:hyperlink
         元素与 HYPERLINK 域代码两种形式）
      5. 引用标记删除（[1]、[1-3]、[注 1]、[来源请求]、[需要解释]、
         ": 508-525" 页码引用等）
      6. 维基百科残留移除（独立成段的"编辑"等 UI 标签）
      7. 空段落移除
      8. 连续重复段落去重
      9. 页眉页脚清除
    """

    ZERO_WIDTH_CHARS = {"\u200b", "\u200c", "\u200d", "\ufeff", "\u2060"}

    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.doc = Document(str(file_path))
        self.stats = {
            "unicode_normalized": 0,
            "whitespace_cleaned": 0,
            "control_chars_removed": 0,
            "hyperlinks_unwrapped": 0,
            "field_hyperlinks_unwrapped": 0,
            "citations_removed": 0,
            "wikipedia_artifacts_removed": 0,
            "empty_paragraphs_removed": 0,
            "duplicate_paragraphs_removed": 0,
            "headers_footers_cleared": 0,
        }

    # ------------------------------------------------------------------ #
    #  各清洗步骤
    # ------------------------------------------------------------------ #

    def normalize_unicode(self):
        for p in self.doc.paragraphs:
            for run in p.runs:
                normalized = unicodedata.normalize("NFC", run.text)
                if normalized != run.text:
                    run.text = normalized
                    self.stats["unicode_normalized"] += 1

    def clean_whitespace(self):
        for p in self.doc.paragraphs:
            for run in p.runs:
                original = run.text
                text = original.replace("\t", " ")
                text = text.replace("\xa0", " ")
                text = text.replace("\u3000", " ")
                text = re.sub(r" {2,}", " ", text)
                text = text.strip()
                if text != original:
                    run.text = text
                    self.stats["whitespace_cleaned"] += 1

    def remove_control_chars(self):
        for p in self.doc.paragraphs:
            for run in p.runs:
                original = run.text
                text = "".join(
                    ch
                    for ch in original
                    if ch not in self.ZERO_WIDTH_CHARS
                    and not (
                        unicodedata.category(ch).startswith("C")
                        and ch not in ("\n", "\r")
                    )
                )
                if text != original:
                    run.text = text
                    removed = len(original) - len(text)
                    self.stats["control_chars_removed"] += removed

    def unwrap_hyperlinks(self):
        body = self.doc.element.body
        hyperlinks = list(body.iter(qn("w:hyperlink")))
        for hyperlink in hyperlinks:
            parent = hyperlink.getparent()
            idx = list(parent).index(hyperlink)
            runs = hyperlink.findall(qn("w:r"))
            for offset, run in enumerate(runs):
                parent.insert(idx + offset, deepcopy(run))
            parent.remove(hyperlink)
            self.stats["hyperlinks_unwrapped"] += 1

    def unwrap_field_hyperlinks(self):
        """解包域代码形式的超链接。

        两种形式：
          1. 简单域: <w:fldSimple w:instr=" HYPERLINK ...">...</w:fldSimple>
             —— 保留内部 runs，移除 fldSimple 包装。
          2. 复杂域: fldChar(begin) -> instrText(HYPERLINK...) ...
             -> fldChar(separate) -> 结果文本 runs -> fldChar(end)
             —— 保留 separate 与 end 之间的显示文本，
                删除指令部分（begin..separate）与 end 标记。
        """
        body = self.doc.element.body

        # 1) 简单域 fldSimple
        for fld in list(body.iter(qn("w:fldSimple"))):
            instr = fld.get(qn("w:instr")) or ""
            if not instr.strip().upper().startswith("HYPERLINK"):
                continue
            parent = fld.getparent()
            idx = list(parent).index(fld)
            runs = fld.findall(qn("w:r"))
            for offset, run in enumerate(runs):
                parent.insert(idx + offset, deepcopy(run))
            parent.remove(fld)
            self.stats["field_hyperlinks_unwrapped"] += 1

        # 2) 复杂域
        for p in body.iter(qn("w:p")):
            self._unwrap_complex_fields_in_paragraph(p)

    def _unwrap_complex_fields_in_paragraph(self, paragraph):
        runs = list(paragraph.iter(qn("w:r")))
        to_remove = set()
        open_fields = []  # 栈: {"begin": run, "hyperlink": bool, "separate": run|None}

        for run in runs:
            for child in run:
                tag = child.tag
                if tag == qn("w:fldChar"):
                    ftype = child.get(qn("w:fldCharType"))
                    if ftype == "begin":
                        open_fields.append(
                            {"begin": run, "hyperlink": False, "separate": None}
                        )
                    elif ftype == "separate":
                        if open_fields and open_fields[-1]["separate"] is None:
                            open_fields[-1]["separate"] = run
                    elif ftype == "end":
                        if not open_fields:
                            continue
                        field = open_fields.pop()
                        if not field["hyperlink"]:
                            continue
                        end_run = run
                        separate = field["separate"]
                        # 删除指令部分: begin .. separate（含），无 separate 时
                        # 整个域（begin .. end）都删除
                        in_removal = False
                        for r in runs:
                            if r is field["begin"]:
                                in_removal = True
                            if in_removal:
                                to_remove.add(r)
                            if separate is not None and r is separate:
                                in_removal = False
                            if r is end_run:
                                break
                        if separate is not None:
                            to_remove.add(end_run)
                        self.stats["field_hyperlinks_unwrapped"] += 1
                elif tag == qn("w:instrText"):
                    if (
                        open_fields
                        and open_fields[-1]["separate"] is None
                        and "HYPERLINK" in (child.text or "").upper()
                    ):
                        open_fields[-1]["hyperlink"] = True

        for run in to_remove:
            run.getparent().remove(run)

    def remove_citations(self):
        citation_re = re.compile(
            r'\[\d+(?:[-,\s]+\d+)*\]'
            r'|\[注\s*\d+\]'
            r'|\[来源请求\]'
            r'|\[需要解释\]'
            r'|[:\uff1a][\u200a\u2009\u202f]+\d+(?:[-\u2013\u2014,，;][\u200a\u2009\u202f ]*\d+)*'
        )
        punct_re = re.compile(r'[.。]{2,}')

        for p in self.doc.paragraphs:
            runs = p.runs
            if not runs:
                continue

            count_before = len(citation_re.findall(p.text))

            for run in runs:
                original = run.text
                cleaned = citation_re.sub("", original)
                cleaned = punct_re.sub("。", cleaned)
                if cleaned != original:
                    run.text = cleaned

            count_after = len(citation_re.findall(p.text))

            if count_after > 0 and len(runs) > 1:
                full_text = "".join(r.text for r in runs)
                cleaned_full = citation_re.sub("", full_text)
                cleaned_full = punct_re.sub("。", cleaned_full)
                runs[0].text = cleaned_full
                for run in runs[1:]:
                    run.text = ""
                count_after = len(citation_re.findall(p.text))

            removed = count_before - count_after
            if removed > 0:
                self.stats["citations_removed"] += removed

    def remove_wikipedia_artifacts(self):
        artifacts = {"编辑", "[编辑]"}
        for p in list(self.doc.paragraphs):
            if p.text.strip() in artifacts:
                p._element.getparent().remove(p._element)
                self.stats["wikipedia_artifacts_removed"] += 1

    def remove_empty_paragraphs(self):
        body = self.doc.element.body
        for p in list(self.doc.paragraphs):
            if not p.text.strip():
                p._element.getparent().remove(p._element)
                self.stats["empty_paragraphs_removed"] += 1

    def remove_duplicate_paragraphs(self):
        seen = set()
        for p in list(self.doc.paragraphs):
            text = p.text.strip()
            if not text:
                continue
            if text in seen:
                p._element.getparent().remove(p._element)
                self.stats["duplicate_paragraphs_removed"] += 1
            else:
                seen.add(text)

    def clear_headers_footers(self):
        for section in self.doc.sections:
            for header in (section.header, section.first_page_header, section.even_page_header):
                if header and header.paragraphs:
                    for p in header.paragraphs:
                        p.clear()
                    self.stats["headers_footers_cleared"] += 1
            for footer in (section.footer, section.first_page_footer, section.even_page_footer):
                if footer and footer.paragraphs:
                    for p in footer.paragraphs:
                        p.clear()
                    self.stats["headers_footers_cleared"] += 1

    # ------------------------------------------------------------------ #
    #  执行与保存
    # ------------------------------------------------------------------ #

    def clean_all(self):
        self.normalize_unicode()
        self.clean_whitespace()
        self.remove_control_chars()
        self.unwrap_hyperlinks()
        self.unwrap_field_hyperlinks()
        self.remove_citations()
        self.remove_wikipedia_artifacts()
        self.remove_empty_paragraphs()
        self.remove_duplicate_paragraphs()
        self.clear_headers_footers()
        return self.stats

    def save(self, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.doc.save(str(output_path))
        return output_path


def clean_file(file_path, output_dir=None):
    file_path = Path(file_path)
    if file_path.suffix.lower() not in (".docx",):
        print(f"跳过非 docx 文件: {file_path.name}")
        return None

    cleaner = DocxCleaner(file_path)
    stats = cleaner.clean_all()

    if output_dir:
        output_dir = Path(output_dir)
        output_path = output_dir / file_path.name
    else:
        output_path = file_path.with_stem(file_path.stem + "_cleaned")

    cleaner.save(output_path)

    print(f"[完成] {file_path.name} -> {output_path.name}")
    for item, count in stats.items():
        if count > 0:
            print(f"    {item}: {count}")
    return output_path


def clean_folder(folder_path, output_dir=None):
    folder = Path(folder_path)
    if not folder.is_dir():
        raise ValueError(f"路径不是文件夹: {folder_path}")

    docx_files = [
        f for f in folder.iterdir()
        if f.suffix.lower() == ".docx" and not f.name.startswith("~$")
    ]
    if not docx_files:
        print(f"文件夹中未找到 docx 文件: {folder_path}")
        return []

    print(f"找到 {len(docx_files)} 个 docx 文件\n")
    results = []
    for f in docx_files:
        result = clean_file(f, output_dir)
        if result:
            results.append(result)
    print(f"\n共清洗 {len(results)} 个文件")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="docx 文件数据清洗工具")
    parser.add_argument("--input_path", default="./data/origin",help="docx 文件路径或包含 docx 文件的文件夹路径，默认读取data/origin文件夹")
    parser.add_argument("--output-dir", default="./data/processed", help="输出目录（默认在/data/processed目录下生成 _cleaned 后缀文件）")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    if input_path.is_dir():
        clean_folder(input_path, args.output_dir)
    elif input_path.is_file():
        clean_file(input_path, args.output_dir)
    else:
        raise ValueError(f"路径不存在: {input_path}")
