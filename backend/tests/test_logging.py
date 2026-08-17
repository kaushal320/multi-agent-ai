from app.agents.logging import log_agent_start, log_agent_success


class TestAgentLogging:
    def test_log_agent_start_returns_float(self):
        state = {"conversation_id": "test-123", "prompt": "hello"}
        t0 = log_agent_start("test_agent", state)
        assert isinstance(t0, float)

    def test_log_agent_success_does_not_raise(self):
        state = {"conversation_id": "test-123"}
        log_agent_success("test_agent", state, 0.0, extra_field="value")
