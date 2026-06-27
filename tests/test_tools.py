"""Unit tests for tool functions (no LLM or ChromaDB needed)."""

import pytest
from src.agent.tools import triage_decision, EMERGENCY_SIGNS, search_cnki


class TestTriageDecision:

    def test_emergency_signal_detected(self):
        result = triage_decision("猫咪两天不吃东西呕吐拉血精神很差")
        assert "🚨 急诊建议" in result
        assert "拉血" in result

    def test_emergency_breathing(self):
        result = triage_decision("猫呼吸困难")
        assert "🚨 急诊建议" in result
        assert "呼吸困难" in result

    def test_urgent_symptoms(self):
        result = triage_decision("猫咪打喷嚏流鼻涕不吃东西")
        assert "⚠️ 建议尽快就医" in result

    def test_mild_observation(self):
        result = triage_decision("猫咪今天打了一次喷嚏")
        assert "ℹ️ 居家观察" in result

    def test_all_emergency_signs_mapped(self):
        """Every emergency sign should trigger EMERGENCY level."""
        for sign in EMERGENCY_SIGNS:
            result = triage_decision(f"猫咪{sign}")
            assert "🚨 急诊建议" in result, f"Sign '{sign}' did not trigger EMERGENCY"

    def test_concern_found_in_urgent(self):
        for sign in ["不吃", "呕吐", "腹泻", "咳嗽", "发热"]:
            result = triage_decision(f"猫咪{sign}")
            assert "就医" in result, f"Concern '{sign}' should trigger at least URGENT"


class TestEmergencySignsIntegrity:

    def test_no_duplicates(self):
        assert len(EMERGENCY_SIGNS) == len(set(EMERGENCY_SIGNS))

    def test_minimal_set(self):
        """Critical signs that must be present."""
        required = ["呼吸困难", "抽搐", "大出血", "中毒", "尿不出来"]
        for s in required:
            assert s in EMERGENCY_SIGNS, f"Missing critical sign: {s}"


class TestSearchCnkiFallback:

    def test_returns_message_when_empty(self):
        """search_cnki should return a formatted message even on API failure."""
        result = search_cnki("猫哮喘")
        assert isinstance(result, str)
        assert len(result) > 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
