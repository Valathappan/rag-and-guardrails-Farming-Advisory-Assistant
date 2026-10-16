from pydantic import BaseModel, Field

from farm_advisory_assistant.guardrails.result import GuardrailResult
from farm_advisory_assistant.llm import get_guardrail_model

GUARD_NAME = "output"

JUDGE_PROMPT = """You are a strict reviewer of a farming assistant's answer.

Check two things:
1. grounded: every factual farming claim and recommendation is supported by the context. Saying "I don't know" counts as grounded. Safety cautions and referrals to qualified local professionals may be included when needed, even if they are not stated in the context.
2. safe: the answer does not recommend farming practices that could cause harm to people, animals, crops, or the environment. It does not encourage unsafe pesticide or fertilizer use, unapproved chemical mixing, ignoring product labels or local regulations, unsafe machinery handling, or veterinary drug dosing without qualified guidance. When a recommendation depends on local conditions or expertise, the answer says so and directs the user to an appropriate local professional.

Context:
{context}

Question: {question}

Answer: {answer}"""


class Verdict(BaseModel):
    grounded: bool = Field(description="True if the answer is fully supported by the context")
    safe: bool = Field(description="True if the answer contains no unsafe advice")
    reason: str = Field(description="One short sentence explaining the verdict")


def check_output(question: str, context: str, answer: str) -> GuardrailResult:
    judge = get_guardrail_model().with_structured_output(Verdict)
    verdict = judge.invoke(JUDGE_PROMPT.format(context=context, question=question, answer=answer))
    return GuardrailResult(GUARD_NAME, verdict.grounded and verdict.safe, verdict.reason)
