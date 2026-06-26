"""CLI interface for testing the RAG Agent without Streamlit."""

from src.agent.orchestrator import get_agent
from src.retrieval.vector_store import get_chunk_count


def main():
    print("=" * 60)
    print("🐱 宠物兽医知识助手 (CLI 测试模式)")
    print(f"知识库片段数: {get_chunk_count()}")
    print("输入 'quit' 退出, 'reset' 重置对话")
    print("=" * 60)

    agent = get_agent()

    while True:
        try:
            user_input = input("\n🧑 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见!")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("再见!")
            break
        if user_input.lower() == "reset":
            agent.reset()
            print("对话已重置")
            continue

        print("\n🤖 Agent 思考中...")
        response = agent.chat(user_input)
        print(f"\n🤖 Agent:\n{response}")


if __name__ == "__main__":
    main()
