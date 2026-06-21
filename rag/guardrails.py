import re
from .schemas import GuardrailResult

INJECTION_PATTERNS = [
    r"ignore (all )?(previous|prior) instructions",
    r"jailbreak",
    r"pretend you are",
    r"developer mode",
    r"reveal.*prompt",
    r"list every chunk",
]

CLINICAL_PATTERNS = [
    r"should i (take|use|stop|start)",
    r"what dose",
    r"dosage for me",
    r"diagnose me",
    r"am i sick",
    r"treat my",
    r"which medicine should i take",
    r"can i take",
    r"is it safe for me",
]

def check_input(question: str) -> GuardrailResult:
    q = (question or "").strip().lower()

    if not q:
        return GuardrailResult(
            allowed=False,
            category="empty",
            reason="Please enter a question.",
        )

    if any(re.search(pattern, q) for pattern in INJECTION_PATTERNS):
        return GuardrailResult(
            allowed=False,
            category="prompt_injection",
            reason="Prompt-injection or jailbreak attempt blocked.",
        )

    if any(re.search(pattern, q) for pattern in CLINICAL_PATTERNS):
        return GuardrailResult(
            allowed=False,
            category="clinical_advice",
            reason="I can summarize biomedical literature, but I cannot provide personal medical advice.",
        )

    # Do not block general research questions like:
    # "summarize this paper", "what are the findings?", "what does it conclude?"
    # Retrieval will decide whether the answer exists in uploaded PDFs.
    return GuardrailResult(allowed=True)