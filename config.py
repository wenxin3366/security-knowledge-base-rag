# config.py
# API 配置
DEEPSEEK_API_KEY = "你的大模型API-KEY"
SILICONFLOW_API_KEY = "硅基流API-KEY"  # 用于 Embedding

# 模型配置
EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"  # 中文 Embedding 模型
CHAT_MODEL = "deepseek-chat"

# 分块配置
CHUNK_SIZE = 500        # 每块字符数
CHUNK_OVERLAP = 50      # 块之间的重叠字符数

# 检索配置
TOP_K = 3               # 每次检索返回的块数

# 对话历史配置
MAX_HISTORY = 10        # 最多保留的对话轮数