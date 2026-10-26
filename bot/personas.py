"""Persona loading from bot/personas/*.md (simple `key: value` frontmatter + prompt body)."""

from dataclasses import dataclass
from pathlib import Path

PERSONAS_DIR = Path(__file__).parent / "personas"
DEFAULT_PERSONA = "granny"
DEFAULT_GEMINI_VOICE = "Kore"


@dataclass(frozen=True)
class Persona:
    id: str
    name: str
    emoji: str
    description: str
    greeting: str
    order: int
    history: int | None
    voice: str
    gemini_voice: str
    length_scale: float
    pitch_cents: float
    tremolo_hz: float
    tremolo_depth: float
    prompt: str

    def public(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "emoji": self.emoji,
            "description": self.description,
        }


def _parse(path: Path) -> Persona:
    text = path.read_text(encoding="utf-8")
    _, front, body = text.split("---", 2)
    meta = {}
    for line in front.strip().splitlines():
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    return Persona(
        id=path.stem,
        name=meta["name"],
        emoji=meta["emoji"],
        description=meta["description"],
        greeting=meta["greeting"],
        order=int(meta["order"]),
        history=int(meta["history"]) if "history" in meta else None,
        voice=meta["voice"],
        gemini_voice=meta.get("gemini_voice", DEFAULT_GEMINI_VOICE),
        length_scale=float(meta.get("length_scale", 1.0)),
        pitch_cents=float(meta.get("pitch_cents", 0)),
        tremolo_hz=float(meta.get("tremolo_hz", 0)),
        tremolo_depth=float(meta.get("tremolo_depth", 0)),
        prompt=body.strip(),
    )


def load_personas() -> dict[str, Persona]:
    personas = [_parse(p) for p in PERSONAS_DIR.glob("*.md")]
    return {p.id: p for p in sorted(personas, key=lambda p: p.order)}
