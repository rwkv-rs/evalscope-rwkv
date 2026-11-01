import re
from typing import Optional

from .content import ContentReasoning


def parse_content_with_reasoning(content: str) -> tuple[str, Optional[ContentReasoning]]:
    """
    Looks for and extracts <think/> tags into reasoning text.

    Returns a tuple:
    - The first element is the input content with the <think> tag and its contents fully removed.
    - The second element is a ContentReasoning object (or None if no <think> tag is found).
    """
    # Match <think> tag with optional attributes anywhere in the string
    pattern = r'<think(?:\s+signature="([^"]*)")?(?:\s+redacted="(true)")?\s*>(.*?)</think>'
    match = re.search(pattern, content, re.DOTALL)

    if match:
        signature = match.group(1)  # This will be None if not present
        redacted_value = match.group(2)  # This will be "true" or None
        reasoning = match.group(3).strip()
        # Remove the matched <think>...</think> from the input
        start, end = match.span()

        return (
            (content[:start] + content[end:]).strip(),
            ContentReasoning(
                reasoning=reasoning,
                signature=signature,
                redacted=redacted_value == 'true',
            ),
        )
    else:
        return content, None


_THINK_OPEN = re.compile(r'<think(?:\s+signature="[^"]*")?(?:\s+redacted="true")?\s*>')
_THINK_CLOSE = '</think>'


def split_final_answer(text: str) -> tuple[str, Optional[str]]:
    """Return the final-answer text of a completion and the reasoning boundary that was applied.

    Closed ``<think>...</think>`` blocks are normally removed when the response is parsed. Two
    shapes survive that parsing and must not be graded as answers:

    * ``'unclosed'``: a ``<think>`` block that never closed (for example a response truncated by
      ``max_tokens``). There is no final answer, so the returned text is what preceded the block.
    * ``'closed_without_opening'``: a ``</think>`` without an opening tag, which happens when the
      chat template places ``<think>`` in the prompt. Only the text after it is the answer.

    Text without either shape is returned unchanged with boundary ``None``.
    """
    opening = None
    for opening in _THINK_OPEN.finditer(text):
        pass
    if opening is not None and _THINK_CLOSE not in text[opening.end():]:
        return text[:opening.start()].rstrip(), 'unclosed'
    if _THINK_CLOSE in text and _THINK_OPEN.search(text) is None:
        return text.rsplit(_THINK_CLOSE, 1)[1].lstrip(), 'closed_without_opening'
    return text, None
