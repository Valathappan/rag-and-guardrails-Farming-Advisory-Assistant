from farm_advisory_assistant.agent.state import AgentState
from farm_advisory_assistant.config import settings
from farm_advisory_assistant.guardrails.nlu_guard import check_input_nlu
from farm_advisory_assistant.guardrails.output_guard import check_output
from farm_advisory_assistant.guardrails.regex_guard import check_input_regex
from farm_advisory_assistant.rag.context_builder import build_context
from farm_advisory_assistant.rag.generator import generate_answer
from farm_advisory_assistant.rag.query_rewriter import rewrite_query
from farm_advisory_assistant.rag.retriever import retrieve

REFUSALS = {
    "regex": "Sorry, I can't process personal or sensitive identifiers, or instructions that try to change how I operate. Please rephrase without them.",
    "nlu": "Sorry, I can only help with safe, farming-related questions.",
    "output": "Sorry, I couldn't verify a safe, reliable answer from the available farming sources.",
}


def regex_guard_node(state: AgentState) -> dict:
    return {"guardrail_results": [check_input_regex(state["question"])]}


def nlu_guard_node(state: AgentState) -> dict:
    return {"guardrail_results": [check_input_nlu(state["question"])]}


def rewrite_query_node(state: AgentState) -> dict:
    return {"search_query": rewrite_query(state["question"])}


def retrieve_node(state: AgentState) -> dict:
    return {"chunks": retrieve(state["search_query"], settings.top_k)}


def build_context_node(state: AgentState) -> dict:
    return {"context": build_context(state["chunks"])}


def generate_node(state: AgentState) -> dict:
    return {"answer": generate_answer(state["question"], state["context"])}


def output_guard_node(state: AgentState) -> dict:
    result = check_output(state["question"], state["context"], state["answer"])
    return {"guardrail_results": [result]}


def refuse_node(state: AgentState) -> dict:
    failed_guard = state["guardrail_results"][-1].guard
    return {"answer": REFUSALS[failed_guard]}


def route_after_guard(state: AgentState) -> str:
    return "continue" if state["guardrail_results"][-1].allowed else "refuse"
