"""
Evaluation Harness for Multi-Agent System

Provides golden-set prompts for testing:
1. Routing accuracy
2. RAG retrieval quality
3. Agent behavior (reflexion, fan-out)
4. End-to-end response quality

All tests use free-tier compatible tools.
"""

import json
import asyncio
from typing import Any
from dataclasses import dataclass, field
from datetime import datetime

# Golden-set test cases for routing accuracy
ROUTING_GOLDEN_SET = [
    # (prompt, expected_agent, description)
    ("Hello, how are you?", "chat", "Simple greeting"),
    ("What's the weather in Tokyo?", "search", "Real-time info needs web search"),
    ("Write a Python function to sort a list", "coding", "Code generation"),
    ("Create a PDF report about Q3 sales", "pdf", "PDF generation"),
    ("Make a PowerPoint about AI trends", "ppt", "PPT generation"),
    ("Generate an image of a sunset", "image", "Image generation"),
    ("What does the uploaded document say about pricing?", "rag", "Document QA"),
    ("Compare the document's claims with current market data", "research_rag", "Document + web search"),
    ("Explain how transformers work", "chat", "General knowledge (no real-time needed)"),
    ("Latest news on OpenAI GPT-5", "search", "Breaking news"),
    ("Debug this Python error: TypeError: 'NoneType'", "coding", "Code debugging"),
    ("Summarize the attached research paper", "rag", "Document summarization"),
    ("Verify the document's statistics against current data", "research_rag", "Fact-checking with docs"),
    ("Write a SQL query for user analytics", "coding", "SQL generation"),
    ("Create a slide deck for investor pitch", "ppt", "Presentation creation"),
]

# Golden-set for reflexion behavior
REFLEXION_GOLDEN_SET = [
    {
        "prompt": "What's the capital of France?",
        "initial_answer": "Paris is the capital of France.",
        "expected_needs_more": False,
        "description": "Complete factual answer",
    },
    {
        "prompt": "What's the current stock price of AAPL?",
        "initial_answer": "I don't have real-time stock data.",
        "expected_needs_more": True,
        "expected_agent": "search",
        "description": "Missing real-time data",
    },
    {
        "prompt": "Summarize the uploaded document",
        "initial_answer": "No documents uploaded yet.",
        "expected_needs_more": True,
        "expected_agent": "rag_research",
        "description": "No document context available",
    },
    {
        "prompt": "Compare the document's claims with latest research",
        "initial_answer": "The document says X, but I haven't checked current research.",
        "expected_needs_more": True,
        "expected_agent": "search",
        "description": "Partial answer needs verification",
    },
]

# Golden-set for fan-out planning
FANOUT_GOLDEN_SET = [
    {
        "prompt": "Search for latest React patterns and write a component",
        "expected_agents": ["search", "coding"],
        "description": "Research + code generation",
    },
    {
        "prompt": "What does the document say and what's the current market?",
        "expected_agents": ["rag_research", "search"],
        "description": "Doc QA + market research",
    },
    {
        "prompt": "Create a PDF report with latest AI statistics",
        "expected_agents": ["search", "pdf"],
        "description": "Data gathering + document generation",
    },
    {
        "prompt": "Write a simple hello world in Python",
        "expected_agents": ["coding"],
        "description": "Pure code generation",
    },
]


@dataclass
class EvalResult:
    """Single evaluation result."""
    test_name: str
    passed: bool
    expected: Any
    actual: Any
    latency_ms: float
    error: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class EvalSuite:
    """Collection of evaluation results."""
    name: str
    results: list[EvalResult] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def total_count(self) -> int:
        return len(self.results)

    @property
    def pass_rate(self) -> float:
        if self.total_count == 0:
            return 0.0
        return self.passed_count / self.total_count

    def add_result(self, result: EvalResult):
        self.results.append(result)

    def summary(self) -> dict:
        return {
            "suite": self.name,
            "total": self.total_count,
            "passed": self.passed_count,
            "failed": self.total_count - self.passed_count,
            "pass_rate": round(self.pass_rate * 100, 1),
            "duration_sec": round((self.completed_at or datetime.now() - self.started_at).total_seconds(), 2) if self.completed_at else None,
        }

    def mark_complete(self):
        self.completed_at = datetime.now()


class RoutingEvaluator:
    """Evaluates router accuracy against golden set."""

    def __init__(self):
        from app.agents.router_agent import router_node
        from app.agents.state import AgentState
        self.router_node = router_node
        self.AgentState = AgentState

    async def evaluate(self) -> EvalSuite:
        suite = EvalSuite("Routing Accuracy")

        for prompt, expected_agent, desc in ROUTING_GOLDEN_SET:
            start = asyncio.get_event_loop().time()

            state = self.AgentState(
                prompt=prompt,
                agent="auto",
                conversation_id=f"eval_{hash(prompt)}",
                request_id="eval",
                ai_response="",
                search_results=[],
                images=[],
                rag_context="",
                rag_sources=[],
                orchestration_plan=[],
                token_usage={},
                iteration=0,
                max_iterations=3,
                reflection="",
                needs_more_info=False,
                handoff=None,
                fanout_agents=[],
            )

            try:
                result = await self.router_node(state)
                actual_agent = result.get("agent", "unknown")
                latency_ms = (asyncio.get_event_loop().time() - start) * 1000

                passed = actual_agent == expected_agent
                suite.add_result(EvalResult(
                    test_name=desc,
                    passed=passed,
                    expected=expected_agent,
                    actual=actual_agent,
                    latency_ms=latency_ms,
                    metadata={"prompt": prompt, "confidence": result.get("reflection", "")}
                ))
            except Exception as e:
                latency_ms = (asyncio.get_event_loop().time() - start) * 1000
                suite.add_result(EvalResult(
                    test_name=desc,
                    passed=False,
                    expected=expected_agent,
                    actual="error",
                    latency_ms=latency_ms,
                    error=str(e),
                    metadata={"prompt": prompt}
                ))

        suite.mark_complete()
        return suite


class FanOutEvaluator:
    """Evaluates fan-out planner decisions."""

    def __init__(self):
        from app.agents.graph import fanout_planner
        from app.agents.state import AgentState
        self.fanout_planner = fanout_planner
        self.AgentState = AgentState

    async def evaluate(self) -> EvalSuite:
        suite = EvalSuite("Fan-Out Planning")

        for case in FANOUT_GOLDEN_SET:
            start = asyncio.get_event_loop().time()

            state = self.AgentState(
                prompt=case["prompt"],
                agent="auto",
                conversation_id=f"eval_{hash(case['prompt'])}",
                request_id="eval",
                ai_response="",
                search_results=[],
                images=[],
                rag_context="",
                rag_sources=[],
                orchestration_plan=[],
                token_usage={},
                iteration=0,
                max_iterations=3,
                reflection="",
                needs_more_info=False,
                handoff=None,
                fanout_agents=[],
            )

            try:
                result = await self.fanout_planner(state)
                actual_agents = set(result.get("fanout_agents", []))
                expected_agents = set(case["expected_agents"])
                latency_ms = (asyncio.get_event_loop().time() - start) * 1000

                # Check if all expected agents are in actual (superset is OK)
                passed = expected_agents.issubset(actual_agents)
                suite.add_result(EvalResult(
                    test_name=case["description"],
                    passed=passed,
                    expected=sorted(expected_agents),
                    actual=sorted(actual_agents),
                    latency_ms=latency_ms,
                    metadata={"prompt": case["prompt"]}
                ))
            except Exception as e:
                latency_ms = (asyncio.get_event_loop().time() - start) * 1000
                suite.add_result(EvalResult(
                    test_name=case["description"],
                    passed=False,
                    expected=sorted(case["expected_agents"]),
                    actual="error",
                    latency_ms=latency_ms,
                    error=str(e),
                    metadata={"prompt": case["prompt"]}
                ))

        suite.mark_complete()
        return suite


class ReflexionEvaluator:
    """Evaluates reflexion self-correction behavior."""

    def __init__(self):
        from app.agents.graph import reflect_node
        from app.agents.state import AgentState
        self.reflect_node = reflect_node
        self.AgentState = AgentState

    async def evaluate(self) -> EvalSuite:
        suite = EvalSuite("Reflexion Loop")

        for case in REFLEXION_GOLDEN_SET:
            start = asyncio.get_event_loop().time()

            state = self.AgentState(
                prompt=case["prompt"],
                agent="chat",
                conversation_id=f"eval_{hash(case['prompt'])}",
                request_id="eval",
                ai_response=case["initial_answer"],
                search_results=[],
                images=[],
                rag_context="",
                rag_sources=[],
                orchestration_plan=[],
                token_usage={},
                iteration=0,
                max_iterations=3,
                reflection="",
                needs_more_info=False,
                handoff=None,
                fanout_agents=[],
            )

            try:
                result = await self.reflect_node(state)
                actual_needs_more = result.get("needs_more_info", False)
                actual_agent = result.get("agent", "none")
                latency_ms = (asyncio.get_event_loop().time() - start) * 1000

                passed = actual_needs_more == case["expected_needs_more"]
                if passed and "expected_agent" in case:
                    passed = actual_agent == case["expected_agent"]

                suite.add_result(EvalResult(
                    test_name=case["description"],
                    passed=passed,
                    expected={"needs_more": case["expected_needs_more"], "agent": case.get("expected_agent", "any")},
                    actual={"needs_more": actual_needs_more, "agent": actual_agent},
                    latency_ms=latency_ms,
                    metadata={"prompt": case["prompt"], "reflection": result.get("reflection", "")}
                ))
            except Exception as e:
                latency_ms = (asyncio.get_event_loop().time() - start) * 1000
                suite.add_result(EvalResult(
                    test_name=case["description"],
                    passed=False,
                    expected={"needs_more": case["expected_needs_more"]},
                    actual="error",
                    latency_ms=latency_ms,
                    error=str(e),
                    metadata={"prompt": case["prompt"]}
                ))

        suite.mark_complete()
        return suite


async def run_all_evaluations() -> dict:
    """Run all evaluation suites and return combined results."""
    print("🧪 Running evaluation suites...")

    evaluators = [
        RoutingEvaluator(),
        FanOutEvaluator(),
        ReflexionEvaluator(),
    ]

    all_suites = []
    for evaluator in evaluators:
        print(f"  Running {evaluator.__class__.__name__}...")
        suite = await evaluator.evaluate()
        all_suites.append(suite)
        print(f"  ✅ {suite.name}: {suite.passed_count}/{suite.total_count} passed ({suite.pass_rate*100:.1f}%)")

    # Combined summary
    total_tests = sum(s.total_count for s in all_suites)
    total_passed = sum(s.passed_count for s in all_suites)

    return {
        "timestamp": datetime.now().isoformat(),
        "total_tests": total_tests,
        "total_passed": total_passed,
        "overall_pass_rate": round(total_passed / total_tests * 100, 1) if total_tests > 0 else 0,
        "suites": [s.summary() for s in all_suites],
        "details": {
            s.name: [r.__dict__ for r in s.results]
            for s in all_suites
        }
    }


if __name__ == "__main__":
    # Run evaluations
    results = asyncio.run(run_all_evaluations())

    # Print summary
    print("\n" + "="*50)
    print("📊 EVALUATION SUMMARY")
    print("="*50)
    print(f"Overall: {results['total_passed']}/{results['total_tests']} ({results['overall_pass_rate']}%)")

    for suite in results["suites"]:
        status = "✅" if suite["passed"] == suite["total"] else "⚠️"
        print(f"  {status} {suite['suite']}: {suite['passed']}/{suite['total']} ({suite['pass_rate']}%)")

    # Save detailed results
    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n💾 Detailed results saved to eval_results.json")