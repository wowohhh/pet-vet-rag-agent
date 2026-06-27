"""Unit tests for context window management."""

import pytest
from src.agent.context import (
    estimate_tokens, estimate_message_tokens, trim_conversation
)


class TestTokenEstimation:

    def test_chinese_text(self):
        assert estimate_tokens("猫咪打喷嚏流鼻涕") > 0
        assert estimate_tokens("") == 1  # minimum 1

    def test_message_tokens(self):
        msgs = [{"role": "user", "content": "猫咪打喷嚏"}]
        assert estimate_message_tokens(msgs) > 0

    def test_empty_content(self):
        msgs = [{"role": "assistant", "content": ""}]
        assert estimate_message_tokens(msgs) >= 0


class TestTrimConversation:

    def test_no_trim_when_under_budget(self):
        msgs = [
            {"role": "system", "content": "You are a vet assistant."},
            {"role": "user", "content": "猫打喷嚏"},
            {"role": "assistant", "content": "可能是上呼吸道感染"},
        ]
        trimmed, summary = trim_conversation(msgs, max_ratio=0.9)
        assert len(trimmed) == len(msgs)
        assert summary == ""

    def test_trims_when_over_budget(self):
        """Build a large conversation that exceeds budget."""
        msgs = [{"role": "system", "content": "sys" * 500}]  # ~250 tokens
        for i in range(30):
            msgs.append({"role": "user", "content": f"第{i}轮" * 100})
            msgs.append({"role": "assistant", "content": f"回答{i}" * 200})
        trimmed, summary = trim_conversation(msgs, max_ratio=0.5)
        assert len(trimmed) < len(msgs)
        assert "历史对话摘要" in summary

    def test_keeps_system_prompt(self):
        msgs = [{"role": "system", "content": "VET_SYSTEM"}, {"role": "user", "content": "x" * 5000}]
        trimmed, _ = trim_conversation(msgs, max_ratio=0.5)
        assert trimmed[0]["role"] == "system"
        assert trimmed[0]["content"] == "VET_SYSTEM"

    def test_summary_includes_trimmed_queries(self):
        msgs = [{"role": "system", "content": "sys" * 500}]
        for i in range(20):
            msgs.append({"role": "user", "content": f"猫咪打喷嚏第{i}天怎么办"})
            msgs.append({"role": "assistant", "content": "建议观察" * 200})
        _, summary = trim_conversation(msgs, max_ratio=0.3)
        assert summary  # should have summary
        assert "打喷嚏" in summary or "猫咪" in summary  # should mention user query


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
