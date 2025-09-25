from __future__ import annotations
import re
from typing import Tuple

# --- Phrases to flag potential emergencies (expand as needed) ---
EMERGENCY_PATTERNS = [
    r"\bchest\s*pain\b",
    r"\bshort(ness)?\s*of\s*breath\b",
    r"\bsuicid(e|al)\b",
    r"\bself[-\s]*harm\b",
    r"\boverdose\b",
    r"\bpoison(ing|ed)?\b",
    r"\b(unconscious|passed\s*out|faint(ing)?)\b",
    r"\bsevere\s*(bleed(ing)?|allergic\s*reaction|anaphylaxis|abdominal\s*pain|headache)\b",
    r"\bstroke\b|\b(face\s*droop|arm\s*weakness|slurred\s*speech)\b",
    r"\b(numb(ness)?|weakness)\s*(on|in)\s*(one|1)\s*(side|arm|leg|face)\b",
]

# Compile once
_EMERG_RE = re.compile("|".join(EMERGENCY_PATTERNS), flags=re.IGNORECASE)

DISCLAIMER = (
    "This assistant shares general health information only and is not a substitute for professional medical advice. "
    "Always consult a qualified clinician for diagnosis or treatment."
)

EMERGENCY_MSG = (
    "If you are experiencing a medical emergency or life-threatening symptoms, call your local emergency number "
    "or go to the nearest emergency department immediately."
)

def screen_safety(user_text: str) -> Tuple[bool, str]:
    """
    Returns (is_emergency, disclaimer_text).
    - is_emergency: True if any emergency pattern is detected.
    - disclaimer_text: General disclaimer; includes emergency guidance when flagged.
    """
    text = (user_text or "").strip()
    is_emerg = bool(_EMERG_RE.search(text))
    disclaimer = DISCLAIMER + (" " + EMERGENCY_MSG if is_emerg else "")
    return is_emerg, disclaimer
