import pytest

from app.agents.models import (
    TokenBudgetExceeded,
    _estimate_tokens,
    check_token_budget,
)


class TestEstimateTokens:
    def test_short_text(self):
        assert _estimate_tokens("hello") == 1

    def test_empty_string(self):
        assert _estimate_tokens("") == 0

    def test_long_text(self):
        text = "a" * 400
        assert _estimate_tokens(text) == 100


class TestTokenBudget:
    def test_within_budget(self):
        check_token_budget("hello world")

    def test_exceeds_budget(self):
        long_prompt = "x" * 20000
        with pytest.raises(TokenBudgetExceeded):
            check_token_budget(long_prompt)

    def test_exceeds_budget_with_context(self):
        prompt = "x" * 10000
        context = "y" * 10000
        with pytest.raises(TokenBudgetExceeded):
            check_token_budget(prompt, context)
