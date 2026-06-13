# file_processor.py
import os
from config import CHUNK_SIZE, CHUNK_OVERLAP


class FileProcessor:
    """文件解析和分块"""

    def process_file(self, file_path):
        """处理文件：读取并分块"""
        if not os.path.exists(file_path):
            return None, f"文件不存在: {file_path}"

        # 获取文件名
        file_name = os.path.basename(file_path)

        # 根据扩展名读取文件
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".txt":
            text = self._read_txt(file_path)
        elif ext == ".pdf":
            text = self._read_pdf(file_path)
        elif ext == ".md":
            text = self._read_txt(file_path)  # Markdown 当作文本读
        else:
            return None, f"不支持的文件类型: {ext}"

        if not text:
            return None, "文件内容为空"

        # 分块
        chunks = self._chunk_text(text)

        return chunks, f"成功处理 {file_name}，共 {len(chunks)} 个知识块"

    def _read_txt(self, file_path):
        """读取 TXT 文件"""
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def _read_pdf(self, file_path):
        """读取 PDF 文件（需要安装 pypdf）"""
        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text
        except ImportError:
            print("⚠️ 请安装 pypdf: pip install pypdf")
            return None

    def _chunk_text(self, text):
        """将长文本分块"""
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + CHUNK_SIZE
            chunk = text[start:end]

            # 清理空白
            chunk = chunk.strip()
            if chunk:
                chunks.append(chunk)

            start = end - CHUNK_OVERLAP

        return chunks