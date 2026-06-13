# vector_store.py
import chromadb
from chromadb.utils import embedding_functions
import hashlib
import uuid
import os
import json


class VectorStore:
    """向量数据库管理 - 支持持久化"""

    def __init__(self, persist_directory="./chroma_db"):
        # 创建持久化客户端（数据会自动保存到磁盘）
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)

        # 创建或获取 collection
        self.collection = self.client.get_or_create_collection(
            name="knowledge_base"
        )

        # 重建文件索引（从数据库读取已有数据）
        self._rebuild_file_index()

        print(f"✅ 向量数据库已初始化，当前存储：{self.collection.count()} 个知识块")

    def _rebuild_file_index(self):
        """从数据库重建文件索引（用于持久化恢复）"""
        self.files = {}

        try:
            # 获取所有数据
            all_data = self.collection.get()

            if all_data and all_data['ids']:
                for i, doc_id in enumerate(all_data['ids']):
                    # 从 ID 中提取文件名（格式：filename_chunk_X）
                    if '_chunk_' in doc_id:
                        file_name = doc_id.split('_chunk_')[0]

                        if file_name not in self.files:
                            self.files[file_name] = {
                                "chunks": 0,
                                "ids": []
                            }

                        self.files[file_name]["chunks"] += 1
                        self.files[file_name]["ids"].append(doc_id)

                print(f"📚 已恢复 {len(self.files)} 个文件的索引")
        except Exception as e:
            print(f"⚠️ 重建索引时出错: {e}")

    def add_document(self, file_name, chunks):
        """添加文档到向量数据库"""
        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            # 生成唯一ID（格式：文件名_chunk_序号）
            chunk_id = f"{file_name}_chunk_{i}"
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append({
                "source": file_name,
                "chunk_index": i,
                "total_chunks": len(chunks)
            })

        # 添加到 Chroma
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

        # 更新文件索引
        self.files[file_name] = {
            "chunks": len(chunks),
            "ids": ids
        }

        print(f"   📦 已存储到数据库: {len(chunks)} 个知识块")
        return len(chunks)

    def search(self, query_text, top_k=3):
        """
        根据查询文本搜索最相似的文本块

        参数:
            query_text: 用户的查询文本
            top_k: 返回的最相似结果数量

        返回:
            相关文本块列表
        """
        if self.collection.count() == 0:
            return []

        try:
            # 使用文本查询（Chroma 会自动处理 Embedding）
            results = self.collection.query(
                query_texts=[query_text],
                n_results=top_k
            )

            # 提取检索到的文本块
            if results and results.get('documents') and results['documents'][0]:
                return results['documents'][0]
            else:
                return []

        except Exception as e:
            print(f"⚠️ 检索失败: {e}")
            return []

    def search_with_metadata(self, query_text, top_k=3):
        """
        搜索并返回详细信息（包含元数据）
        """
        if self.collection.count() == 0:
            return []

        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=top_k
            )

            formatted_results = []
            if results and results.get('documents') and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results.get('metadatas') else {}
                    formatted_results.append({
                        "text": doc,
                        "source": metadata.get("source", "未知"),
                        "chunk_index": metadata.get("chunk_index", 0)
                    })

            return formatted_results

        except Exception as e:
            print(f"⚠️ 检索失败: {e}")
            return []

    def list_files(self):
        """列出所有已上传的文件"""
        return list(self.files.keys())

    def get_file_info(self, file_name):
        """获取指定文件的详细信息"""
        return self.files.get(file_name, None)

    def delete_file(self, file_name):
        """删除指定文件的所有块"""
        if file_name not in self.files:
            return False

        ids_to_delete = self.files[file_name]["ids"]

        try:
            self.collection.delete(ids=ids_to_delete)
            del self.files[file_name]
            return True
        except Exception as e:
            print(f"❌ 删除失败: {e}")
            return False

    def get_stats(self):
        """获取统计信息"""
        return {
            "total_chunks": self.collection.count(),
            "files": len(self.files),
            "file_list": list(self.files.keys()),
            "persist_directory": self.persist_directory
        }

    def clear_all(self):
        """清空所有数据（慎用）"""
        try:
            # 删除所有数据
            all_ids = self.collection.get()['ids']
            if all_ids:
                self.collection.delete(ids=all_ids)
            self.files = {}
            print("✅ 已清空所有数据")
            return True
        except Exception as e:
            print(f"❌ 清空失败: {e}")
            return False
