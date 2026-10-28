"""The granny from bot/persona.md (simple `key: value` frontmatter + prompt body)."""

from dataclasses import dataclass
from pathlib import Path

PERSONA_FILE = Path(__file__).parent / "persona.md"


@dataclass(frozen=True)
class Persona:
    name: str
    greeting: str
    prompt: str


def load_persona() -> Persona:
    text = PERSONA_FILE.read_text(encoding="utf-8")
    _, front, body = text.split("---", 2)
    meta = {}
    for line in front.strip().splitlines():
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    return Persona(name=meta["name"], greeting=meta["greeting"], prompt=body.strip())
