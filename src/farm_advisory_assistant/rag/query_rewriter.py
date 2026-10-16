from farm_advisory_assistant.llm import get_agent_model

REWRITE_PROMPT = """Rewrite the user's question as a concise search query for a farming knowledge base.
Preserve important details such as crop, location, growth stage, symptoms, and conditions.
Expand abbreviations only when their meaning is clear; otherwise keep them unchanged.
Remove conversational filler, but do not remove details that could affect search results.
Do not answer the question or add information that the user did not provide.
Return only the search query.

Question: {question}"""


def rewrite_query(question: str) -> str:
    response = get_agent_model().invoke(REWRITE_PROMPT.format(question=question))
    return response.text.strip().strip('"')
