"""Alien Manual tool generator (FR-004, NG-3).

Every tool is built from a template: an archetype (what the tool does), a
domain vocabulary and seeded random names. A tool may carry a ``twist`` that
makes its documentation contradict the habit its name suggests:

* ``flip_args``  - ``move(src, dst)`` documented as src = destination.
* ``invert_op``  - ``increase`` documented as lowering the quantity.
* ``odd_unit``   - a value documented in tenths/hundredths/thousandths of a unit.
* ``rename``     - the doc says to call a different tool name.

Public API (all JSON-serialisable through ``to_dict`` / ``from_dict``):

* ``Call``: ``{"name": str, "arguments": dict}``.
* ``Param(name, type, description)``; ``type`` is ``"string"`` or ``"integer"``.
* ``Tool(name, domain, archetype, params, doc, twist, call_name, unit_factor)``.
  ``name`` is the habit name shown in the doc heading; ``call_name`` is the
  name the oracle calls (differs from ``name`` only for ``rename``).
* ``Task(id, tool, request, oracle_call, habit_call, is_trap)``.
* ``generate_tools(rng, domains, twist_pattern, per_domain, used_nouns, hardness)``
* ``generate_tasks(tools, n, rng, split)``

For every twisted task ``habit_call != oracle_call``; for a normal task they are
equal. ``hardness`` only changes doc text (a hook for E-1): at 0 the twist is a
plain "Note:" line; at 1 or more it is folded into the parameter descriptions
and ``hardness`` benign extra notes are added, so the twist is harder to spot.
Calls, requests and ids do not depend on ``hardness``.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

Call = dict[str, Any]

TWISTS: tuple[str, ...] = ("flip_args", "invert_op", "odd_unit", "rename")
ARCHETYPES: tuple[str, ...] = ("transfer", "adjust", "set", "fetch")
# Archetype each twist needs; rename works on any archetype.
_TWIST_ARCHETYPE = {"flip_args": "transfer", "invert_op": "adjust", "odd_unit": "set"}


@dataclass(frozen=True)
class Domain:
    """Vocabulary template for a domain (not a tool)."""

    thing: str
    place: str
    quantity: str
    unit: str


DOMAINS: dict[str, Domain] = {
    "inventory": Domain("crate", "bay", "stock level", "unit"),
    "scheduling": Domain("shift", "slot", "planned duration", "minute"),
    "fluids": Domain("valve", "tank", "flow volume", "litre"),
    "thermal": Domain("furnace", "zone", "temperature", "degree"),
    "finance": Domain("ledger", "account", "balance", "credit"),
    "navigation": Domain("beacon", "sector", "range", "metre"),
    "messaging": Domain("packet", "channel", "payload size", "byte"),
    "audio": Domain("track", "deck", "gain", "decibel"),
    "botany": Domain("seedling", "bed", "water allowance", "gram"),
    "power": Domain("cell", "rack", "output", "watt"),
    "storage": Domain("archive", "vault", "quota", "block"),
    "logistics": Domain("pallet", "dock", "load", "kilogram"),
}

HABIT_VERBS: dict[str, tuple[str, ...]] = {
    "transfer": ("move", "shift", "relocate", "carry"),
    "adjust": ("increase", "raise", "boost", "grow"),
    "set": ("set", "assign", "tune", "fix"),
    "fetch": ("read", "fetch", "lookup", "inspect"),
}
ALIAS_VERBS: tuple[str, ...] = (
    "pull", "retrieve", "obtain", "query", "grab", "acquire", "collect", "summon",
)

_TRANSFER_PAIRS = (("src", "dst"), ("source", "target"), ("from_id", "to_id"), ("origin", "dest"))
_ITEM_NAMES = ("item", "entry", "subject", "ref")
_DELTA_NAMES = ("delta", "amount", "change")
_VALUE_NAMES = ("value", "level", "setting", "reading")
_FRACTIONS = {10: "tenths of a", 100: "hundredths of a", 1000: "thousandths of a"}
_BENIGN_NOTES = (
    "Note: calls are written to the audit log.",
    "Note: requests are processed in the order they arrive.",
    "Note: identifiers are case sensitive.",
    "Note: the tool returns immediately and does not wait for confirmation.",
    "Note: unknown identifiers are rejected without side effects.",
    "Note: at most 100 calls per minute are accepted.",
)

_ONSETS = ("b", "d", "g", "k", "m", "n", "p", "r", "s", "t", "v", "z", "qu", "th")
_VOWELS = ("a", "e", "i", "o", "u")
_CODAS = ("", "k", "n", "r", "x", "m")


@dataclass(frozen=True)
class Param:
    name: str
    type: str
    description: str

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "type": self.type, "description": self.description}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Param:
        return cls(d["name"], d["type"], d["description"])


@dataclass(frozen=True)
class Tool:
    name: str
    domain: str
    archetype: str
    params: tuple[Param, ...]
    doc: str
    twist: str | None
    call_name: str
    unit_factor: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "domain": self.domain,
            "archetype": self.archetype,
            "params": [p.to_dict() for p in self.params],
            "doc": self.doc,
            "twist": self.twist,
            "call_name": self.call_name,
            "unit_factor": self.unit_factor,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Tool:
        return cls(
            name=d["name"],
            domain=d["domain"],
            archetype=d["archetype"],
            params=tuple(Param.from_dict(p) for p in d["params"]),
            doc=d["doc"],
            twist=d["twist"],
            call_name=d["call_name"],
            unit_factor=d["unit_factor"],
        )


@dataclass(frozen=True)
class Task:
    id: str
    tool: Tool
    request: str
    oracle_call: Call
    habit_call: Call
    is_trap: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tool": self.tool.to_dict(),
            "request": self.request,
            "oracle_call": self.oracle_call,
            "habit_call": self.habit_call,
            "is_trap": self.is_trap,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Task:
        return cls(
            id=d["id"],
            tool=Tool.from_dict(d["tool"]),
            request=d["request"],
            oracle_call=d["oracle_call"],
            habit_call=d["habit_call"],
            is_trap=d["is_trap"],
        )


def _invented_word(rng: random.Random) -> str:
    return "".join(
        rng.choice(_ONSETS) + rng.choice(_VOWELS) + rng.choice(_CODAS)
        for _ in range(rng.choice((2, 3)))
    )


def _a(word: str) -> str:
    return ("an " if word[0] in "aeiou" else "a ") + word


def _plural(word: str, n: int) -> str:
    return word if n == 1 else word + "s"


def _param_descriptions(archetype: str, twist: str | None, subtle: bool, dom: Domain,
                        factor: int) -> list[str]:
    """Descriptions for the non-item params, in order. ``subtle`` folds the twist in."""
    if archetype == "transfer":
        if not subtle:
            return [f"A {dom.place} identifier."] * 2
        start = f"The {dom.place} the {dom.thing} starts from."
        end = f"The {dom.place} the {dom.thing} ends up in."
        return [end, start] if twist == "flip_args" else [start, end]
    if archetype == "adjust":
        if not subtle:
            return [f"Signed change to the {dom.quantity}."]
        up, down = "raise", "lower"
        if twist == "invert_op":
            up, down = down, up
        return [f"Signed change to the {dom.quantity}; positive values {up} it, "
                f"negative values {down} it."]
    if archetype == "set":
        if not subtle:
            return [f"The new {dom.quantity}."]
        if twist == "odd_unit":
            return [f"The new {dom.quantity}, in {_FRACTIONS[factor]} {dom.unit} "
                    f"({factor} per whole {dom.unit})."]
        return [f"The new {dom.quantity}, in whole {dom.unit}s."]
    return []


def _build_doc(rng: random.Random, name: str, call_name: str, archetype: str,
               twist: str | None, dom: Domain, params: tuple[Param, ...], factor: int,
               hardness: int) -> str:
    summary = {
        "transfer": f"Moves {_a(dom.thing)} between two {dom.place}s.",
        "adjust": f"Changes the {dom.quantity} of {_a(dom.thing)} by a signed amount.",
        "set": f"Sets the {dom.quantity} of {_a(dom.thing)}.",
        "fetch": f"Returns the stored record for {_a(dom.thing)}.",
    }[archetype]
    lines = [summary]
    notes: list[str] = []
    explicit = {
        "flip_args": lambda: (
            f"Note: `{params[1].name}` is the {dom.place} the {dom.thing} ends up in and "
            f"`{params[2].name}` is the {dom.place} it starts from, the reverse of what "
            "the names suggest."),
        "invert_op": lambda: (
            f"Note: despite its name, a positive `{params[1].name}` lowers the "
            f"{dom.quantity} and a negative `{params[1].name}` raises it."),
        "odd_unit": lambda: (
            f"Note: `{params[1].name}` is in {_FRACTIONS[factor]} {dom.unit}, so send "
            f"{factor} for each whole {dom.unit}."),
        "rename": lambda: (
            f"Note: this tool is registered as `{call_name}`; the name `{name}` is "
            f"retired and rejected. Always call `{call_name}`."),
    }
    if hardness == 0:
        notes.append(explicit[twist]() if twist else rng.choice(_BENIGN_NOTES))
    else:
        if twist == "rename":
            lines.append(f"Registered as `{call_name}`; the older name `{name}` is retired "
                         "and rejected.")
        notes.extend(rng.sample(_BENIGN_NOTES, min(hardness, len(_BENIGN_NOTES))))
    lines.extend(notes)
    lines.append("Parameters:")
    lines.extend(f"- {p.name} ({p.type}): {p.description}" for p in params)
    return "\n".join(lines)


def _make_tool(rng: random.Random, domain: str, twist: str | None, used_nouns: set[str],
               hardness: int) -> Tool:
    dom = DOMAINS[domain]
    archetype = _TWIST_ARCHETYPE.get(twist) or rng.choice(ARCHETYPES)
    noun = _invented_word(rng)
    while noun in used_nouns:
        noun = _invented_word(rng)
    used_nouns.add(noun)
    name = f"{rng.choice(HABIT_VERBS[archetype])}_{noun}"
    call_name = f"{rng.choice(ALIAS_VERBS)}_{noun}" if twist == "rename" else name
    item = _ITEM_NAMES[rng.randrange(len(_ITEM_NAMES))]
    if archetype == "transfer":
        a, b = rng.choice(_TRANSFER_PAIRS)
        names, types = [item, a, b], ["string"] * 3
    elif archetype == "adjust":
        names, types = [item, rng.choice(_DELTA_NAMES)], ["string", "integer"]
    elif archetype == "set":
        names, types = [item, rng.choice(_VALUE_NAMES)], ["string", "integer"]
    else:
        names, types = [item], ["string"]
    factor = rng.choice(tuple(_FRACTIONS)) if twist == "odd_unit" else 1
    item_desc = {
        "transfer": f"The {dom.thing} to move.",
        "adjust": f"The {dom.thing} to change.",
        "set": f"The {dom.thing} to update.",
        "fetch": f"The {dom.thing} to look up.",
    }[archetype]
    descs = [item_desc, *_param_descriptions(archetype, twist, hardness > 0, dom, factor)]
    params = tuple(Param(n, t, d) for n, t, d in zip(names, types, descs))
    doc_rng = random.Random(f"doc:{noun}")  # hardness never shifts the structural draws above
    doc = _build_doc(doc_rng, name, call_name, archetype, twist, dom, params, factor, hardness)
    return Tool(name, domain, archetype, params, doc, twist, call_name, factor)


def generate_tools(rng: random.Random, domains: Sequence[str],
                   twist_pattern: Sequence[str | None], per_domain: int,
                   used_nouns: set[str], hardness: int = 0) -> list[Tool]:
    """Generate ``per_domain`` tools per domain; tool ``i`` (global index) gets
    ``twist_pattern[i % len(twist_pattern)]``. ``used_nouns`` is shared across
    calls so tool names never collide between splits."""
    tools = []
    for domain in domains:
        for _ in range(per_domain):
            twist = twist_pattern[len(tools) % len(twist_pattern)]
            tools.append(_make_tool(rng, domain, twist, used_nouns, hardness))
    return tools


def _make_task(tool: Tool, rng: random.Random, task_id: str) -> Task:
    dom = DOMAINS[tool.domain]
    names = [p.name for p in tool.params]
    item = f"{dom.thing[0].upper()}-{rng.randint(10, 99)}"
    args: dict[str, Any] = {names[0]: item}
    if tool.archetype == "transfer":
        n1, n2 = rng.sample(range(1, 10), 2)
        p1, p2 = f"{dom.place}-{n1}", f"{dom.place}-{n2}"
        request = rng.choice((
            f"Move the {dom.thing} {item} from {p1} to {p2}.",
            f"Please relocate {dom.thing} {item}: it is in {p1} and needs to go to {p2}.",
        ))
        habit_args = {**args, names[1]: p1, names[2]: p2}
        oracle_args = ({**args, names[1]: p2, names[2]: p1} if tool.twist == "flip_args"
                       else habit_args)
    elif tool.archetype == "adjust":
        n = rng.randint(1, 50)
        raising = rng.random() < 0.5
        verb = "Raise" if raising else "Lower"
        request = f"{verb} the {dom.quantity} of {dom.thing} {item} by {n}."
        habit_args = {**args, names[1]: n if raising else -n}
        oracle_args = ({**args, names[1]: -habit_args[names[1]]} if tool.twist == "invert_op"
                       else habit_args)
    elif tool.archetype == "set":
        n = rng.randint(1, 99)
        request = f"Set the {dom.quantity} of {dom.thing} {item} to {n} {_plural(dom.unit, n)}."
        habit_args = {**args, names[1]: n}
        oracle_args = {**args, names[1]: n * tool.unit_factor}
    else:
        request = rng.choice((
            f"Look up the record for {dom.thing} {item}.",
            f"Show me the details of {dom.thing} {item}.",
        ))
        habit_args = oracle_args = args
    return Task(
        id=task_id,
        tool=tool,
        request=request,
        oracle_call={"name": tool.call_name, "arguments": oracle_args},
        habit_call={"name": tool.name, "arguments": habit_args},
        is_trap=tool.twist is not None,
    )


def generate_tasks(tools: Sequence[Tool], n: int, rng: random.Random, split: str) -> list[Task]:
    """``n`` tasks spread evenly over ``tools``, shuffled, with ids ``<split>-0000``..."""
    per_tool, extra = divmod(n, len(tools))
    pending = [tool for i, tool in enumerate(tools) for _ in range(per_tool + (i < extra))]
    rng.shuffle(pending)
    return [_make_task(tool, rng, f"{split}-{i:04d}") for i, tool in enumerate(pending)]
