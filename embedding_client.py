# embedding_client.py
import requests
from config import SILICONFLOW_API_KEY, EMBEDDING_MODEL


class EmbeddingClient:
    """Embedding API 客户端"""

    def __init__(self):
        self.api_key = SILICONFLOW_API_KEY
        self.url = "https://api.siliconflow.cn/v1/embeddings"
        self.model = EMBEDDING_MODEL

    def get_embedding(self, text):
        """将文本转换成向量"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "input": text
        }

        response = requests.post(self.url, headers=headers, json=payload)

        if response.status_code == 200:
            return response.json()["data"][0]["embedding"]
        else:
            print(f"⚠️ Embedding API 错误: {response.status_code}")
            return None

    def get_embeddings_batch(self, texts):
        """批量将文本转换成向量"""
        return [self.get_embedding(text) for text in texts if text]