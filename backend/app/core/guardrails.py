import re
from typing import Any

try:
    import logfire

    LOGFIRE_AVAILABLE = True
except ImportError:
    LOGFIRE_AVAILABLE = False


# Known dangerous or jailbreak pattern triggers
JAILBREAK_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"disregard\s+(all\s+)?prior\s+rules",
    r"system\s+prompt\s+override",
    r"you\s+are\s+now\s+in\s+dan\s+mode",
]

# Unsafe cyber-attack and exploitation patterns (handles typos like "how di i hack u", "hack you", "hack system")
HACKING_SECURITY_PATTERNS = [
    r"\bhack\s+(u|you|your|this|the|a|system|app|server|bot|ai|website|database|cluster|node)\b",
    r"how\s+\w+\s+(i\s+)?(hack|exploit|breach|crack|compromise|penetrate)",
    r"\b(hack|exploit|breach|crack|penetrate)\b\s+(into|system|server|network|cluster|database|auth|site)",
    r"\bbypass\b\s+(auth|authentication|password|firewall|waf|security|login|restriction)",
    r"(sql\s+injection|reverse\s+shell|buffer\s+overflow|ddos)\s+(payload|script|code|attack)",
    r"generate\s+(malware|ransomware|trojan|keylogger|exploit|virus|rootkit)",
]

# Output patterns that should never appear in responses
DANGEROUS_OUTPUT_PATTERNS = [
    r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
    r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",  # Credit card
    r"BEGIN\s+(RSA\s+)?PRIVATE\s+KEY",  # Private keys
]

# Fast-path greetings mapping to save LLM tokens completely
FAST_PATH_GREETINGS = {
    "hi": "Hello! How can I assist you with code, research, or documents today?",
    "hello": "Hello there! How can I help you today?",
    "hey": "Hey! What can I help you build or explore today?",
    "hi there": "Hi there! Feel free to ask me anything.",
    "hello there": "Hello! How can I assist you today?",
    "good morning": "Good morning! How can I help you today?",
    "good afternoon": "Good afternoon! How can I help you today?",
    "good evening": "Good evening! How can I help you today?",
}


class GuardrailViolationError(Exception):
    def __init__(self, message: str, category: str = "safety"):
        super().__init__(message)
        self.category = category


def check_fast_path_greeting(prompt: str) -> str | None:
    """Returns a direct response for simple greetings to save 100% of LLM tokens."""
    cleaned = re.sub(r"[^\w\s]", "", prompt.strip().lower())
    return FAST_PATH_GREETINGS.get(cleaned)


def validate_input_prompt(prompt: str) -> dict[str, Any]:
    """Validates input prompt against Guardrails AI rules and records Logfire spans."""
    prompt_clean = prompt.strip()

    if LOGFIRE_AVAILABLE:
        with logfire.span(
            "guardrails_input_inspection", prompt_length=len(prompt_clean)
        ):
            return _perform_input_checks(prompt_clean)
    else:
        return _perform_input_checks(prompt_clean)


def _perform_input_checks(prompt: str) -> dict[str, Any]:
    # Check for empty prompt
    if not prompt:
        raise GuardrailViolationError("Prompt cannot be empty.", category="validation")

    # Check for prompt injection / jailbreak patterns
    for pattern in JAILBREAK_PATTERNS:
        if re.search(pattern, prompt, re.IGNORECASE):
            if LOGFIRE_AVAILABLE:
                logfire.warn(
                    "Guardrails VIOLATION: Prompt injection attempt", pattern=pattern
                )
            raise GuardrailViolationError(
                "Security policy violation: Prompt injection attempt detected.",
                category="prompt_injection",
            )

    # Check for malicious hacking / exploit patterns
    for pattern in HACKING_SECURITY_PATTERNS:
        if re.search(pattern, prompt, re.IGNORECASE):
            if LOGFIRE_AVAILABLE:
                logfire.warn(
                    "Guardrails VIOLATION: Hacking / Exploit request", pattern=pattern
                )
            raise GuardrailViolationError(
                "Security policy violation: Requests for active hacking instructions or exploitation are restricted.",
                category="cybersecurity",
            )

    if LOGFIRE_AVAILABLE:
        logfire.info(
            "Guardrails Input Security Check PASSED", prompt_length=len(prompt)
        )

    return {"status": "passed", "prompt": prompt}


def validate_output_response(response: str) -> dict[str, Any]:
    """Validates agent generated response content against Guardrails AI rules."""
    if LOGFIRE_AVAILABLE:
        with logfire.span(
            "guardrails_output_inspection", response_length=len(response)
        ):
            _perform_output_checks(response)
            logfire.info(
                "Guardrails Output Content Inspection PASSED",
                response_length=len(response),
            )
    else:
        _perform_output_checks(response)
    return {"status": "passed", "response": response}


def _perform_output_checks(response: str) -> None:
    """Check output for dangerous content patterns."""
    for pattern in DANGEROUS_OUTPUT_PATTERNS:
        if re.search(pattern, response, re.IGNORECASE):
            raise GuardrailViolationError(
                "Output policy violation: Response contains sensitive data pattern.",
                category="output_safety",
            )
