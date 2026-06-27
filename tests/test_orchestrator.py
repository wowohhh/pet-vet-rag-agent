"""Unit tests for orchestrator parsing functions (no LLM needed)."""

import pytest
from src.agent.orchestrator import (
    _extract_citations, _extract_triage, _determine_source, Citation, TriageInfo
)


class TestExtractCitations:

    def test_local_format(self):
        result = """### 文献 1 (相关度: 0.85)
**标题**: 猫上呼吸道感染研究
**期刊**: 中国兽医杂志 (2018)
**摘录**: 本研究调查了306例猫上呼吸道感染病例，发现主要病原体包括猫疱疹病毒和杯状病毒。
"""
        citations = _extract_citations([{"name": "search_knowledge_base", "result": result}])
        assert len(citations) == 1
        assert citations[0].title == "猫上呼吸道感染研究"
        assert "中国兽医杂志" in citations[0].journal
        assert citations[0].year == "2018"

    def test_empty_title_fallback_to_journal(self):
        result = """### 文献 1 (相关度: 0.01)
**标题**:
**期刊**: Veterinary Focus (2017)
**摘录**: 菌、花粉等清除时出现的一种无意识的反射表现形式。
"""
        citations = _extract_citations([{"name": "search_knowledge_base", "result": result}])
        assert len(citations) >= 1
        # Should fallback to journal as title
        assert "Veterinary Focus" in citations[0].title

    def test_multiple_results(self):
        result = """### 文献 1
**标题**: 论文A
**期刊**: 期刊A (2020)
**摘录**: 内容A
### 文献 2
**标题**: 论文B
**期刊**: 期刊B (2021)
**摘录**: 内容B
"""
        citations = _extract_citations([{"name": "search_knowledge_base", "result": result}])
        assert len(citations) == 2

    def test_cnki_web_result_format(self):
        result = """## 🌐 知网检索结果（来自网络搜索，非本地知识库）

### 文献 1
**标题**: 猫传染性腹膜炎研究进展
**摘要**: 猫传染性腹膜炎是由猫冠状病毒突变引起的高度致死性疾病。
"""
        citations = _extract_citations([{"name": "search_cnki", "result": result}])
        assert len(citations) >= 1
        assert "猫传染性腹膜炎" in citations[0].title
        assert "知网" in citations[0].journal

    def test_empty_result(self):
        citations = _extract_citations([{"name": "search_knowledge_base", "result": "未找到相关文献。"}])
        assert citations == []

    def test_no_tool_results(self):
        assert _extract_citations([]) == []


class TestExtractTriage:

    def test_emergency_from_tool_result(self):
        tool_results = [{"name": "triage_decision",
                         "result": "## 🚨 急诊建议\n\n检测到危险信号「**拉血**」，建议**立即**前往最近的宠物医院急诊科。"}]
        triage = _extract_triage(tool_results, "")
        assert triage.level == "EMERGENCY"
        assert "拉血" in triage.signal

    def test_urgent_from_tool_result(self):
        tool_results = [{"name": "triage_decision",
                         "result": "## ⚠️ 建议尽快就医\n\n检测到以下症状: 不吃, 呕吐, 腹泻"}]
        triage = _extract_triage(tool_results, "")
        assert triage.level == "URGENT"

    def test_observe_from_tool_result(self):
        tool_results = [{"name": "triage_decision",
                         "result": "## ℹ️ 居家观察\n\n未检测到明显的急诊或紧急症状。"}]
        triage = _extract_triage(tool_results, "")
        assert triage.level == "OBSERVE"

    def test_fallback_to_answer_scan(self):
        """When triage_decision not called, scan answer for emergency keywords."""
        triage = _extract_triage([], "猫咪出现抽搐，建议立即就医")
        assert triage.level == "EMERGENCY"

    def test_unknown_when_no_info(self):
        triage = _extract_triage([], "猫咪今天很好")
        assert triage.level == "UNKNOWN"


class TestDetermineSource:

    def test_local(self):
        assert _determine_source([
            {"name": "search_knowledge_base", "result": "### 文献 1\n**标题**: 某论文"}
        ]) == "local"

    def test_cnki(self):
        assert _determine_source([
            {"name": "search_knowledge_base", "result": "未找到相关文献。"},
            {"name": "search_cnki", "result": "## 🌐 知网检索结果\n### 文献 1\n**标题**: 某论文"}
        ]) == "cnki"

    def test_fallback(self):
        assert _determine_source([
            {"name": "search_knowledge_base", "result": "未找到相关文献。"},
        ]) == "fallback"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
