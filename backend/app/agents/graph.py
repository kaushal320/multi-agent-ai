import asyncio
import json

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.agents.chat_agent import chat_node
from app.agents.coding_agent import coding_node
from app.agents.image_agent import image_node
from app.agents.models import get_model
from app.agents.pdf_agent import pdf_node
from app.agents.ppt_agent import ppt_node
from app.agents.rag_agent import rag_research_node
from app.agents.router_agent import router_node
from app.agents.search_agent import search_node
from app.agents.state import AgentState
from app.core.observability import obs


def _route(state: AgentState) -> str:
    obs.info("Routing decision", agent=state.get("agent", "chat"))
    return state.get("agent", "chat")


def _after_search(state: AgentState) -> str:
    return "end" if state.get("ai_response") else "chat"


def _after_rag(state: AgentState) -> str:
    if state.get("ai_response"):
        return "end"
    return "search" if state.get("agent") == "research_rag" else "chat"


async def parallel_research(state: AgentState) -> dict:
    """Run RAG retrieval and web search in parallel for research_rag agent."""
    with obs.span("parallel_research_execution", conversation_id=state.get("conversation_id", "unknown")):
        rag_task = rag_research_node(state)
        search_task = search_node(state)
        rag_result, search_result = await asyncio.gather(rag_task, search_task)
        obs.parallel_execution(
            conversation_id=state.get("conversation_id", "unknown"),
            tasks=["rag_research", "web_search"],
        )
        # Merge results
        merged = {**rag_result, **search_result}
        return merged


# ============================================================
# Dynamic Fan-Out: Run multiple agents in parallel
# ============================================================

# Registry of available agent nodes for fan-out
# Each entry: agent_name -> (node_function, output_type)
# output_type: "context" = returns context for chat synthesis (search, rag_research)
# output_type: "final" = returns final ai_response directly (coding, pdf, ppt, image)
FANOUT_NODES = {
    "search": (search_node, "context"),
    "rag_research": (rag_research_node, "context"),
    "coding": (coding_node, "final"),
    "pdf": (pdf_node, "final"),
    "ppt": (ppt_node, "final"),
    "image": (image_node, "final"),
}


async def fanout_planner(state: AgentState) -> dict:
    """
    LLM-based planner that decides which agents to run in parallel
    based on the user's query and current context.
    """
    prompt = state.get("prompt", "")
    conversation_id = state.get("conversation_id", "unknown")

    # Build agent descriptions dynamically from registry
    agent_descriptions = []
    for agent_name, (node_func, output_type) in FANOUT_NODES.items():
        if output_type == "context":
            agent_descriptions.append(f"- {agent_name}: Gathers context/evidence for synthesis (returns search_results, rag_context, etc.)")
        else:
            agent_descriptions.append(f"- {agent_name}: Produces final answer directly (returns ai_response, images)")

    planner_prompt = f"""Given the user's question, determine which agents should run in PARALLEL to gather comprehensive information.

Available agents:
{chr(10).join(agent_descriptions)}

User Question: {prompt}

Return ONLY a JSON array of agent names to run in parallel. Examples:
- ["search"] - only web search needed
- ["rag_research"] - only document search needed
- ["search", "rag_research"] - both needed (comparison, verification)
- ["coding"] - code generation task
- ["pdf"] - PDF document generation
- ["ppt"] - PowerPoint presentation generation
- ["image"] - Image generation
- ["search", "coding"] - search for APIs then write code
- [] - no parallel agents needed (direct answer sufficient)

Choose agents that complement each other. Be concise. Prefer context-gathering agents for questions, final-output agents for generation tasks."""

    with obs.span("fanout_planner", conversation_id=conversation_id):
        model = get_model("router")
        try:
            result = await model.ainvoke([("human", planner_prompt)])
            response_text = result.content if isinstance(result.content, str) else str(result.content)

            # Parse JSON array
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0].strip()
            else:
                json_str = response_text.strip()

            agents = json.loads(json_str)

            # Validate agents
            valid_agents = [a for a in agents if a in FANOUT_NODES]

            obs.info("Fanout plan created",
                     conversation_id=conversation_id,
                     requested_agents=agents,
                     valid_agents=valid_agents)

            return {"fanout_agents": valid_agents}

        except Exception as e:
            obs.error("Fanout planner failed", conversation_id=conversation_id, error=e)
            return {"fanout_agents": []}

    with obs.span("fanout_planner", conversation_id=conversation_id):
        model = get_model("router")
        try:
            result = await model.ainvoke([("human", planner_prompt)])
            response_text = result.content if isinstance(result.content, str) else str(result.content)

            # Parse JSON array
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0].strip()
            else:
                json_str = response_text.strip()

            agents = json.loads(json_str)

            # Validate agents
            valid_agents = [a for a in agents if a in FANOUT_NODES]

            obs.info("Fanout plan created",
                     conversation_id=conversation_id,
                     requested_agents=agents,
                     valid_agents=valid_agents)

            return {"fanout_agents": valid_agents}

        except Exception as e:
            obs.error("Fanout planner failed", conversation_id=conversation_id, error=e)
            return {"fanout_agents": []}


async def fanout_executor(state: AgentState) -> dict:
    """Execute the planned agents in parallel and merge results."""
    fanout_agents = state.get("fanout_agents", [])
    conversation_id = state.get("conversation_id", "unknown")

    if not fanout_agents:
        return {}

    with obs.span("fanout_execution",
                  conversation_id=conversation_id,
                  agents=fanout_agents):

        # Create tasks for each agent
        tasks = [FANOUT_NODES[agent][0](state) for agent in fanout_agents]

        # Execute in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Merge results based on output type
        merged = {}
        has_final_output = False

        for agent, result in zip(fanout_agents, results):
            if isinstance(result, Exception):
                obs.error(f"Fanout agent {agent} failed", conversation_id=conversation_id, error=result)
                continue
            if not isinstance(result, dict):
                continue

            output_type = FANOUT_NODES[agent][1]

            if output_type == "final":
                # Final output agents (coding, pdf, ppt, image) - use their ai_response directly
                # But only if we don't already have a final answer
                if "ai_response" in result and result["ai_response"] and not has_final_output:
                    merged["ai_response"] = result["ai_response"]
                    if "images" in result:
                        merged["images"] = result["images"]
                    has_final_output = True
            else:
                # Context agents (search, rag_research) - merge context for synthesis
                for key in ("search_results", "rag_context", "rag_sources", "token_usage"):
                    if key in result and result[key]:
                        if key in merged:
                            # For lists, extend; for strings, append
                            if isinstance(merged[key], list) and isinstance(result[key], list):
                                merged[key].extend(result[key])
                            elif isinstance(merged[key], str) and isinstance(result[key], str):
                                merged[key] += "\n\n" + result[key]
                            else:
                                merged[key] = result[key]
                        else:
                            merged[key] = result[key]

        obs.parallel_execution(
            conversation_id=conversation_id,
            tasks=fanout_agents,
        )

        return merged


# ============================================================
# Reflexion / Self-Correction Loop
# ============================================================

REFLEXION_PROMPT = """You are a quality evaluator for an AI assistant. Given the user's question, the current answer, and available evidence, determine if the answer is complete and accurate.

Return ONLY a JSON object with this exact structure:
{
  "needs_more_info": boolean,
  "reason": "brief explanation of what is missing or why the answer is sufficient",
  "suggested_agent": "search" | "rag_research" | "chat" | "none"
}

EVALUATION CRITERIA:
- If the answer says "I don't know" or "not in context" → needs_more_info: true, suggested_agent: "search" or "rag_research"
- If the answer is a direct factual claim without evidence → needs_more_info: true
- If the question asks for current/recent info and no web search was done → needs_more_info: true, suggested_agent: "search"
- If the question references uploaded documents and no RAG was done → needs_more_info: true, suggested_agent: "rag_research"
- If the answer is complete, well-supported, and directly addresses the question → needs_more_info: false, suggested_agent: "none"

User Question: {prompt}

Current Answer: {answer}

Available Evidence:
- RAG Context: {rag_context}
- Search Results: {search_results}
- Agent Used: {agent}
- Iteration: {iteration} of {max_iterations}

Be strict but fair. Prefer "needs_more_info: false" when the answer genuinely addresses the question."""


async def reflect_node(state: AgentState) -> dict:
    """Self-critique: evaluate if the current answer is sufficient or needs more research."""
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 3)

    # If we've hit max iterations, force completion
    if iteration >= max_iterations:
        obs.info("Reflexion: Max iterations reached, forcing completion",
                 conversation_id=state.get("conversation_id", "unknown"),
                 iteration=iteration, max_iterations=max_iterations)
        return {
            "needs_more_info": False,
            "reflection": f"Max iterations ({max_iterations}) reached. Accepting current answer.",
            "agent": "chat",  # Route to chat for final synthesis
        }

    current_answer = state.get("ai_response", "")
    rag_context = state.get("rag_context", "")[:1000]  # Truncate for token budget
    search_results = state.get("search_results", [])
    search_summary = "\n".join(search_results[:3]) if search_results else "None"
    agent_used = state.get("agent", "unknown")
    prompt = state.get("prompt", "")

    with obs.span("reflexion_evaluation",
                  conversation_id=state.get("conversation_id", "unknown"),
                  iteration=iteration,
                  agent=agent_used):

        model = get_model("router")  # Use fast/cheap model for evaluation

        eval_prompt = REFLEXION_PROMPT.format(
            prompt=prompt,
            answer=current_answer[:2000] if current_answer else "(no answer yet)",
            rag_context=rag_context,
            search_results=search_summary,
            agent=agent_used,
            iteration=iteration + 1,
            max_iterations=max_iterations,
        )

        try:
            result = await model.ainvoke([("human", eval_prompt)])
            # Parse JSON from response
            response_text = result.content if isinstance(result.content, str) else str(result.content)

            # Extract JSON (handle potential markdown code blocks)
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0].strip()
            else:
                json_str = response_text.strip()

            decision = json.loads(json_str)

            needs_more = decision.get("needs_more_info", False)
            reason = decision.get("reason", "No reason provided")
            suggested = decision.get("suggested_agent", "none")

            # Determine next agent
            if not needs_more or suggested == "none":
                next_agent = "chat"
                needs_more = False
            else:
                next_agent = suggested

            obs.info("Reflexion decision",
                     conversation_id=state.get("conversation_id", "unknown"),
                     iteration=iteration + 1,
                     needs_more_info=needs_more,
                     reason=reason,
                     suggested_agent=suggested,
                     next_agent=next_agent)

            return {
                "iteration": iteration + 1,
                "needs_more_info": needs_more,
                "reflection": reason,
                "agent": next_agent,
            }

        except Exception as e:
            obs.error("Reflexion evaluation failed, defaulting to completion",
                      conversation_id=state.get("conversation_id", "unknown"),
                      error=e)
            # On error, don't loop infinitely - force completion
            return {
                "iteration": iteration + 1,
                "needs_more_info": False,
                "reflection": f"Reflexion error: {type(e).__name__}. Accepting current answer.",
                "agent": "chat",
            }


def _after_reflection(state: AgentState) -> str:
    """Route after reflexion: continue loop or end."""
    if state.get("needs_more_info", False):
        agent = state.get("agent", "chat")
        # Only route to valid agents that can gather more info
        if agent in ("search", "rag_research", "parallel_research"):
            return agent
    return "end"


builder = StateGraph(AgentState)
builder.add_node("router", router_node)
builder.add_node("search", search_node)
builder.add_node("chat", chat_node)
builder.add_node("coding", coding_node)
builder.add_node("pdf", pdf_node)
builder.add_node("ppt", ppt_node)
builder.add_node("image", image_node)
builder.add_node("rag_research", rag_research_node)
builder.add_node("parallel_research", parallel_research)
builder.add_node("reflect", reflect_node)
builder.add_node("fanout_planner", fanout_planner)
builder.add_node("fanout_executor", fanout_executor)
builder.add_edge(START, "router")

# Router: everything goes through fanout planner EXCEPT direct generation agents
# Generation agents (coding, pdf, ppt, image) still go direct for backward compat
# but fanout_planner can ALSO dispatch them for complex multi-agent tasks
builder.add_conditional_edges(
    "router",
    _route,
    {
        "chat": "fanout_planner",
        "search": "fanout_planner",
        "coding": "coding",
        "pdf": "pdf",
        "ppt": "ppt",
        "image": "image",
        "rag": "fanout_planner",
        "research_rag": "parallel_research",
    },
)

# Fanout planner -> executor
builder.add_edge("fanout_planner", "fanout_executor")

# Fanout executor routes based on what was executed:
# - If only context agents (search, rag) -> chat for synthesis
# - If final-output agents (coding, pdf, ppt, image) -> reflect directly
# - If mixed -> chat for synthesis (context wins)
def _after_fanout(state: AgentState) -> str:
    """Route after fanout execution."""
    fanout_agents = state.get("fanout_agents", [])
    if not fanout_agents:
        return "chat"  # No agents planned, go to chat for direct answer

    # Check if any final-output agents were run
    has_final = any(FANOUT_NODES[a][1] == "final" for a in fanout_agents if a in FANOUT_NODES)
    has_context = any(FANOUT_NODES[a][1] == "context" for a in fanout_agents if a in FANOUT_NODES)

    if has_final and not has_context:
        # Only final-output agents (e.g., just coding) -> go to reflect
        return "reflect"
    else:
        # Context agents (search, rag) or mixed -> chat for synthesis
        return "chat"

builder.add_conditional_edges(
    "fanout_executor",
    _after_fanout,
    {
        "chat": "chat",
        "reflect": "reflect",
    },
)

# Direct agents (coding, pdf, ppt, image) go to reflect
builder.add_edge("coding", "reflect")
builder.add_edge("pdf", "reflect")
builder.add_edge("ppt", "reflect")
builder.add_edge("image", "reflect")
# Search and rag_research go to reflect after fanout
builder.add_edge("search", "reflect")
builder.add_edge("rag_research", "reflect")
# Parallel research goes to chat for synthesis, then reflect
builder.add_edge("parallel_research", "chat")
# Chat goes to reflect for self-correction
builder.add_edge("chat", "reflect")
# Reflexion decides: continue loop or end
builder.add_conditional_edges(
    "reflect",
    _after_reflection,
    {
        "search": "fanout_planner",  # Loop back through planner
        "rag_research": "fanout_planner",
        "parallel_research": "parallel_research",
        "end": END,
    },
)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)
