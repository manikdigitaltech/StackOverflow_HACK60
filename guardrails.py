import re
import logging
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger("paper_reviewer.guardrails")

# ==========================================
# 1. ADVERSARIAL PATTERNS & SANITIZATION
# ==========================================

# Signatures for typical indirect prompt injection strings
ADVERSARIAL_PATTERNS = [
    r"(?i)ignore\s+(all\s+)?prior\s+instructions",
    r"(?i)ignore\s+(all\s+)?previous\s+instructions",
    r"(?i)system\s+override",
    r"(?i)attention\s+reviewer",
    r"(?i)you\s+must\s+output\s+a\s+perfect\s+score",
    r"(?i)override\s+system\s+scoring",
    r"(?i)skip\s+(all\s+)?tasks",
    r"(?i)bypass\s+evaluation",
    r"(?i)format\s+output\s+as\s+follows\s+and\s+ignore\s+constraints"
]

COMPILED_ADVERSARIAL_RE = re.compile("|".join(ADVERSARIAL_PATTERNS))

def sanitize_pdf_text(text: str) -> Tuple[str, bool]:
    """
    Cleans raw strings from the PDF, strips invisible obfuscation characters,
    and defangs adversarial prompt injection strings.
    
    Returns:
        (sanitized_text, flag_triggered)
    """
    if not text:
        return "", False
    
    # Strip zero-width spacing variants often used to break tokenizer pattern matching
    clean_text = re.sub(r'[\u200B-\u200D\uFEFF\u200E\u200F]', '', text)
    
    # Check for adversarial command sequences
    if COMPILED_ADVERSARIAL_RE.search(clean_text):
        logger.warning("Guardrail Triggered: Adversarial prompt injection pattern stripped.")
        # Defang by substituting the text with a systemic alert string
        clean_text = COMPILED_ADVERSARIAL_RE.sub(
            "[SECURITY ALERT: ADVERSARIAL PROMPT INJECTION DETECTED AND EXCISED BY GUARDRAIL LAYERS]", 
            clean_text
        )
        return clean_text, True
        
    return clean_text, False


# ==========================================
# 2. ADVERSARIAL WRAPPING (STRUCTURAL SYNTAX)
# ==========================================

def format_secure_payload(tag_name: str, raw_content: str) -> str:
    """
    Wraps sanitized variables into isolated XML tags.
    This acts as a structural defense mechanism inside system prompts.
    """
    sanitized, _ = sanitize_pdf_text(raw_content)
    # Ensure nested fake tags within the paper can't break out of the enclosure
    escaped_content = sanitized.replace(f"</{tag_name}>", f"[MALICIOUS CLOSING TAG TRIED TO BREAK OUT]")
    
    return f"<{tag_name}>\n{escaped_content}\n</{tag_name}>"


# ==========================================
# 3. OUTPUT GUARDRAILS (POST-LLM VERIFICATION)
# ==========================================

def verify_output_safety(llm_output: str) -> bool:
    """
    Evaluates generated output strings to ensure the local model wasn't hijacked 
    into leaking system text or generating restricted conversational text.
    """
    if not llm_output:
        return False
    
    # If the local model suddenly prints systemic instruction headers or leaks instructions
    leaks_instructions = "You are an expert ICLR/NeurIPS area chair" in llm_output or "output strictly as JSON" in llm_output
    if leaks_instructions:
        logger.error("Post-LLM Guardrail Failure: Model leaked internal prompt patterns.")
        return False
        
    return True


# ==========================================
# 4. REBUTTAL PROCESSING SYNTAX
# ==========================================

def prepare_rebuttal_payload(initial_assessments: List[Dict[str, Any]], author_rebuttal: str) -> Dict[str, Any]:
    """
    Sanitizes and structuralizes both the historical pipeline assessments 
    and the raw author rebuttal string into clean dictionary variables for 
    the meta-review loop context.
    """
    # Wrap the user-provided author rebuttal securely
    secure_rebuttal = format_secure_payload("author_rebuttal_submission", author_rebuttal)
    
    # Convert past JSON assessments into data tracking context blocks
    review_context_blocks = []
    for assessment in initial_assessments:
        agent_name = assessment.get("agent_name", "unknown_agent")
        output_json = assessment.get("output_json", "")
        review_context_blocks.append(f"--- Initial {agent_name} Assessment ---\n{output_json}")
        
    unified_review_history = "\n".join(review_context_blocks)
    
    return {
        "historical_reviews": f"<historical_reviews>\n{unified_review_history}\n</historical_reviews>",
        "author_rebuttal": secure_rebuttal
    }