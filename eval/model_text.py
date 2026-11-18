"""Reject CJK at model text boundaries without rewriting or echoing input."""

# Include CJK punctuation, kana, Hangul and the unified ideograph extensions.
CJK_RANGES = (
    (0x1100, 0x11FF), (0x2E80, 0x9FFF), (0xA960, 0xA97F),
    (0xAC00, 0xD7FF), (0xF900, 0xFAFF), (0xFE30, 0xFE4F),
    (0xFF65, 0xFFDC), (0x16FE0, 0x18DFF), (0x1AFF0, 0x1B2FF),
    (0x1F200, 0x1F2FF), (0x20000, 0x3FFFF),
)


def cjk_count(text: str) -> int:
    return sum(any(first <= ord(char) <= last for first, last in CJK_RANGES)
               for char in text)


def require_english_model_text(text: str) -> str:
    """Fail before sending; retain all approved English characters verbatim."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError('model text must be a nonempty string')
    if cjk_count(text):
        raise ValueError('model text contains CJK characters; English text is required')
    return text
