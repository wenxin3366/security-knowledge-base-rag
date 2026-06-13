# main.py
import sys
from embedding_client import EmbeddingClient
from vector_store import VectorStore
from file_processor import FileProcessor
from chat_client import ChatClient
from config import TOP_K


class RAGCLI:
    """命令行 RAG 知识库系统"""

    def __init__(self):
        self.embedding_client = EmbeddingClient()
        self.vector_store = VectorStore()
        self.file_processor = FileProcessor()
        self.chat_client = ChatClient()

    def run(self):
        """运行主程序"""
        self._print_banner()

        while True:
            try:
                user_input = input("\n📝 你: ").strip()

                if not user_input:
                    continue

                # 解析命令
                if user_input.startswith("/"):
                    self._handle_command(user_input)
                else:
                    # 普通输入当作问题
                    self._handle_question(user_input)

            except KeyboardInterrupt:
                print("\n\n👋 再见！")
                break
            except Exception as e:
                print(f"❌ 错误: {e}")

    def _print_banner(self):
        """打印欢迎界面"""
        banner = """
╔══════════════════════════════════════════════════════════════╗
║                    📚 个人知识库问答系统                      ║
║                         (命令行版)                           ║
╠══════════════════════════════════════════════════════════════╣
║  命令：                                                      ║
║    /upload <文件路径>   - 上传文件到知识库                   ║
║    /list               - 查看已上传的文件                   ║
║    /stats              - 查看知识库统计                     ║
║    /delete <文件名>     - 删除指定文件                       ║
║    /history            - 查看对话历史                       ║
║    /clear              - 清空对话历史                       ║
║    /help               - 显示帮助                           ║
║    /exit               - 退出程序                           ║
╠══════════════════════════════════════════════════════════════╣
║  直接输入问题即可开始问答                                    ║
╚══════════════════════════════════════════════════════════════╝
        """
        print(banner)

    def _handle_command(self, command):
        """处理命令"""
        parts = command.split()
        cmd = parts[0].lower()

        if cmd == "/upload":
            if len(parts) < 2:
                print("❌ 用法: /upload <文件路径>")
                return
            file_path = parts[1]
            self._upload_file(file_path)

        elif cmd == "/list":
            self._list_files()

        elif cmd == "/stats":
            self._show_stats()

        elif cmd == "/delete":
            if len(parts) < 2:
                print("❌ 用法: /delete <文件名>")
                return
            file_name = parts[1]
            self._delete_file(file_name)

        elif cmd == "/history":
            self._show_history()

        elif cmd == "/clear":
            result = self.chat_client.clear_history()
            print(result)

        elif cmd == "/help":
            self._print_banner()

        elif cmd == "/exit":
            print("👋 再见！")
            sys.exit(0)

        else:
            print(f"❌ 未知命令: {cmd}，输入 /help 查看帮助")

    def _upload_file(self, file_path):
        """上传文件到知识库"""
        print(f"📄 正在处理: {file_path}")

        # 处理文件
        chunks, message = self.file_processor.process_file(file_path)

        if chunks is None:
            print(f"❌ {message}")
            return

        print(f"   📊 {message}")

        # 获取文件名
        import os
        file_name = os.path.basename(file_path)

        # 生成向量并存储
        print("   🔄 正在生成向量...")

        documents = []
        embeddings = []

        for i, chunk in enumerate(chunks):
            print(f"      处理第 {i + 1}/{len(chunks)} 块...")
            embedding = self.embedding_client.get_embedding(chunk)
            if embedding:
                documents.append(chunk)
                embeddings.append(embedding)

        # 存储到向量数据库
        chunk_count = self.vector_store.add_document(file_name, documents)
        print(f"   ✅ 已存储 {chunk_count} 个知识块到数据库")

    def _list_files(self):
        """列出已上传的文件"""
        files = self.vector_store.list_files()

        if not files:
            print("📭 知识库为空，请使用 /upload 上传文件")
        else:
            print("\n📁 已上传的文件:")
            for f in files:
                print(f"   • {f}")

    def _show_stats(self):
        """显示统计信息"""
        stats = self.vector_store.get_stats()
        print(f"""
📊 知识库统计:
   • 知识块总数: {stats['total_chunks']}
   • 文件数量: {stats['files']}
   • 文件列表: {', '.join(stats['file_list']) if stats['file_list'] else '无'}
        """)

    def _delete_file(self, file_name):
        """删除文件"""
        result = self.vector_store.delete_file(file_name)
        if result:
            print(f"✅ 已删除文件: {file_name}")
        else:
            print(f"❌ 文件不存在: {file_name}")

    def _show_history(self):
        """显示对话历史"""
        history = self.chat_client.get_history()

        if not history:
            print("📭 暂无对话历史")
        else:
            print("\n📜 对话历史:")
            for msg in history:
                role = "👤 我" if msg["role"] == "user" else "🤖 AI"
                content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
                print(f"   {role}: {content}")

    def _handle_question(self, question):
        """处理用户问题"""
        print("🔍 正在检索相关知识...")

        # 直接使用文本检索（Chroma 内部会自动调用 Embedding）
        # 不需要手动获取 query_embedding
        retrieved_chunks = self.vector_store.search(question, TOP_K)

        if not retrieved_chunks:
            print("📭 知识库中没有找到相关信息")
            # 仍然尝试回答，但告诉用户没有找到资料
            answer = self.chat_client.ask_without_context(question)
            print(f"\n🤖 {answer}")
            return

        print(f"   📚 找到 {len(retrieved_chunks)} 个相关段落")

        # 显示检索来源（可选）
        # for i, chunk in enumerate(retrieved_chunks):
        #     preview = chunk[:80] + "..." if len(chunk) > 80 else chunk
        #     print(f"      段落 {i+1}: {preview}")

        # 调用 AI 回答
        print("💭 AI 正在思考...")
        answer = self.chat_client.ask_with_context(question, retrieved_chunks)

        print(f"\n🤖 {answer}")


def main():
    """主函数入口"""
    cli = RAGCLI()
    cli.run()


if __name__ == "__main__":
    main()