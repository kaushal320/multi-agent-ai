from app.agents.router_agent import ROUTER_AGENTS, ROUTER_SYSTEM_PROMPT


class TestRouterConfig:
    def test_all_agents_listed(self):
        expected = {
            "chat",
            "search",
            "coding",
            "pdf",
            "ppt",
            "image",
            "rag",
            "research_rag",
        }
        assert set(ROUTER_AGENTS) == expected

    def test_system_prompt_mentions_all_agents(self):
        for agent in ROUTER_AGENTS:
            assert agent in ROUTER_SYSTEM_PROMPT.lower(), (
                f"Agent '{agent}' not mentioned in router system prompt"
            )


class TestAgentState:
    def test_initial_state_has_all_fields(self):
        import typing

        from app.agents.state import AgentState

        annotations = typing.get_type_hints(AgentState)
        required_fields = {
            "prompt",
            "agent",
            "conversation_id",
            "request_id",
            "ai_response",
            "search_results",
            "images",
            "rag_context",
            "rag_sources",
            "orchestration_plan",
            "token_usage",
        }
        assert required_fields == set(annotations.keys())


class TestGraphStructure:
    def test_graph_compiles(self):
        from app.agents.graph import graph

        assert graph is not None

    def test_graph_has_expected_nodes(self):
        from app.agents.graph import builder

        node_names = set(builder.nodes.keys())
        expected = {
            "router",
            "search",
            "chat",
            "coding",
            "pdf",
            "ppt",
            "image",
            "rag_research",
        }
        assert expected == node_names
