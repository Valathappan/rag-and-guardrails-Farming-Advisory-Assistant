import re

from farm_advisory_assistant.guardrails.result import GuardrailResult

GUARD_NAME = "regex"
MAX_INPUT_LENGTH = 500

BLOCKED_PATTERNS = {
    "prompt injection": re.compile(
        r"ignore (all |any )?(the |your )?(previous|prior|above) (instructions|rules|prompts?)"
        r"|disregard (all |any )?(the |your )?(instructions|rules)"
        r"|(reveal|show|print) (me )?(your |the )?(system|hidden) prompt"
        r"|you are now\b"
        r"|jailbreak",
        re.IGNORECASE,
    ),
    "email address": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "phone number": re.compile(r"(?<!\d)(\+?\d{1,3}[\s-]?)?\d{10}(?!\d)"),
    "card number": re.compile(r"(?<!\d)(\d{4}[\s-]?){3}\d{4}(?!\d)"),
    "farm identifier": re.compile(
        r"\b(?:farm(?:er)?|agri(?:culture)?)[\s_-]*"
        r"(?:(?:account|registration|beneficiary|policy)[\s_-]*(?:number|no\.?|id|#)|id)"
        r"\s*[:#-]?\s*[A-Z0-9][A-Z0-9/-]{3,}\b",
        re.IGNORECASE,
    ),
}


def check_input_regex(text: str) -> GuardrailResult:
    if len(text) > MAX_INPUT_LENGTH:
        return GuardrailResult(GUARD_NAME, False, f"input longer than {MAX_INPUT_LENGTH} characters")

    for name, pattern in BLOCKED_PATTERNS.items():
        if pattern.search(text):
            return GuardrailResult(GUARD_NAME, False, f"{name} detected")

    return GuardrailResult(GUARD_NAME, True, "no blocked pattern found")
