"""Emotion/audio tags in LLM replies: "[sighs]" is acted by expressive TTS, never read or shown."""

import re

# Square tags of expressive TTS ("[sighs]", "[short pause]") and sidecar tokens ("<|emotion:anger|>").
_TAG_RE = re.compile(r"\[[^\]]*\]\s*|<\|[^|]*\|>\s*")
# Stage directions in the old style ("<кричит>", "<пауза>"): no TTS engine understands them.
_ANGLE_TAG_RE = re.compile(r"<[^<>\n]*>")
# An unclosed "[" held back this long is plain text, not a tag.
_MAX_TAG_LEN = 40


def strip_tags(text: str) -> str:
    """Remove square, sidecar and angle tags: for engines that cannot act them out."""
    return _ANGLE_TAG_RE.sub("", _TAG_RE.sub("", text))


def strip_angle_tags(text: str) -> str:
    """Remove only angle tags: for engines that understand "[...]" tags."""
    return _ANGLE_TAG_RE.sub("", text)


class TagStreamStripper:
    """strip_tags for text that arrives in word-sized frames.

    TTS word timestamps split "[short pause]" into "[short" and "pause]", so a tag can span frames.
    """

    def __init__(self):
        self._pending = ""

    def reset(self) -> None:
        self._pending = ""

    def feed(self, chunk: str) -> str:
        """Returns the visible part of `chunk`; an unclosed tag is held until it closes."""
        text = strip_tags(self._pending + chunk)
        self._pending = ""
        open_at = text.find("[")
        if open_at == -1:
            return text
        if len(text) - open_at > _MAX_TAG_LEN:
            return text
        self._pending = text[open_at:]
        return text[:open_at]
