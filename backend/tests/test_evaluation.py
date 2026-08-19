"""
Pytest-compatible evaluation tests.
Run with: pytest backend/tests/test_evaluation.py -v
"""
import pytest
import asyncio

from tests.eval_harness import (
    run_all_evaluations,
    RoutingEvaluator,
    FanOutEvaluator,
    ReflexionEvaluator,
)


class TestRoutingAccuracy:
    """Test router classification accuracy against golden set."""

    @pytest.mark.asyncio
    async def test_routing_golden_set(self):
        evaluator = RoutingEvaluator()
        suite = await evaluator.evaluate()

        # Assert minimum pass rate (adjust as needed)
        assert suite.pass_rate >= 0.7, f"Routing accuracy too low: {suite.pass_rate*100:.1f}%"

        # Check specific critical routes
        critical_routes = ["Simple greeting", "Code generation", "Document QA", "Real-time info needs web search"]
        for result in suite.results:
            if result.test_name in critical_routes:
                assert result.passed, f"Critical route failed: {result.test_name} (expected {result.expected}, got {result.actual})"


class TestFanOutPlanning:
    """Test fan-out planner decisions."""

    @pytest.mark.asyncio
    async def test_fanout_golden_set(self):
        evaluator = FanOutEvaluator()
        suite = await evaluator.evaluate()

        # Assert minimum pass rate
        assert suite.pass_rate >= 0.7, f"Fan-out planning accuracy too low: {suite.pass_rate*100:.1f}%"


class TestReflexionLoop:
    """Test reflexion self-correction behavior."""

    @pytest.mark.asyncio
    async def test_reflexion_golden_set(self):
        evaluator = ReflexionEvaluator()
        suite = await evaluator.evaluate()

        # Assert minimum pass rate
        assert suite.pass_rate >= 0.7, f"Reflexion accuracy too low: {suite.pass_rate*100:.1f}%"


class TestFullEvaluationSuite:
    """Run complete evaluation suite."""

    @pytest.mark.asyncio
    async def test_full_evaluation(self):
        """Run all evaluations and ensure overall quality."""
        results = await run_all_evaluations()

        # Overall pass rate should be reasonable
        assert results["overall_pass_rate"] >= 65, f"Overall pass rate too low: {results['overall_pass_rate']}%"

        # Each suite should have at least some passing tests
        for suite in results["suites"]:
            assert suite["passed"] > 0, f"Suite {suite['suite']} had 0 passing tests"

        # Print results for CI visibility
        print(f"\n📊 Evaluation Results:")
        print(f"   Overall: {results['total_passed']}/{results['total_tests']} ({results['overall_pass_rate']}%)")
        for suite in results["suites"]:
            print(f"   {suite['suite']}: {suite['passed']}/{suite['total']} ({suite['pass_rate']}%)")


if __name__ == "__main__":
    # Allow running directly
    asyncio.run(run_all_evaluations())