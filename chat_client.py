# chat_client.py
import requests
from config import DEEPSEEK_API_KEY, CHAT_MODEL, TOP_K


class ChatClient:
    """Chat API 客户端，支持上下文记忆"""

    def __init__(self):
        self.api_key = DEEPSEEK_API_KEY
        self.url = "https://api.deepseek.com/chat/completions"
        self.model = CHAT_MODEL
        self.messages = []  # 对话历史

    def ask_with_context(self, question, retrieved_chunks):
        """基于检索结果和上下文回答问题"""

        # 构建知识上下文
        context = "\n\n---\n\n".join(retrieved_chunks)

        # 构建系统提示
        system_prompt = f"""
                你是一个知识库问答助手。请根据以下参考资料回答用户的问题。
                【参考资料】
                {context}
                
                要求：
                1. 只根据参考资料回答，不要使用外部知识
                2. 如果参考资料中没有相关信息，请说"根据现有知识库无法回答这个问题"
                3. 回答要简洁、准确、专业
        """

        # 构建消息列表
        messages = [
            {"role": "system", "content": system_prompt}
        ]

        # 添加历史对话（保留最近几轮）
        messages.extend(self.messages[-10:])

        # 添加当前问题
        messages.append({"role": "user", "content": question})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 500,
            "temperature": 0.3
        }

        response = requests.post(self.url, headers=headers, json=payload)

        if response.status_code == 200:
            answer = response.json()["choices"][0]["message"]["content"]

            # 保存对话历史
            self.messages.append({"role": "user", "content": question})
            self.messages.append({"role": "assistant", "content": answer})

            return answer
        else:
            return f"API 调用失败: {response.status_code}"

    def clear_history(self):
        """清空对话历史"""
        self.messages = []
        return "✅ 对话历史已清空"

    def get_history(self):
        """获取对话历史"""
        return self.messages


    # chat_client.py 中添加这个方法,实现持久化
    def ask_without_context(self, question):
        """没有检索到知识时的回答"""
        messages = [
            {"role": "system", "content": "你是一个知识库问答助手。知识库中没有找到相关信息，请礼貌地告诉用户这个情况。"},
        ]

        # 添加历史对话
        messages.extend(self.messages[-10:])
        messages.append({"role": "user", "content": question})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 300,
            "temperature": 0.3
        }

        response = requests.post(self.url, headers=headers, json=payload)

        if response.status_code == 200:
            answer = response.json()["choices"][0]["message"]["content"]
            self.messages.append({"role": "user", "content": question})
            self.messages.append({"role": "assistant", "content": answer})
            return answer
        else:
            return f"API 调用失败: {response.status_code}"