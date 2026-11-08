"""Train / dev / held-out splits with a leak check (FR-001, R-2, B-2).

Public API:

* ``make_splits(seed, hardness=0, tools_per_domain=8) -> Splits``. The 12
  domains are shuffled by seed into 6 train, 3 held-out and 3 dev domains
  (dev is a separate pool so tuning against dev never peeks at held-out).
  One twist family is held out per seed: train has none of it, held-out has
  only that family plus untwisted normal tools. Sizes: ``TRAIN_SIZE`` 600,
  ``HELDOUT_SIZE`` 240, ``DEV_SIZE`` 100. ``hardness`` is the E-1 hook; it
  only changes tool docs (see ``alien.generator``).
* ``Splits(train, dev, heldout, heldout_twist, seed, hardness)``.
* ``assert_no_leak(splits)`` raises ``LeakError``; ``make_splits`` calls it.
* ``splits_to_json(splits) -> str`` / ``splits_from_json(text) -> Splits``.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass

from alien.generator import DOMAINS, TWISTS, Task, generate_tasks, generate_tools

TRAIN_SIZE = 600
HELDOUT_SIZE = 240
DEV_SIZE = 100
_N_TRAIN_DOMAINS = 6
_N_HELDOUT_DOMAINS = 3


class LeakError(Exception):
    """A held-out tool, domain or twist family reached train, or dev overlaps another split."""


@dataclass(frozen=True)
class Splits:
    train: list[Task]
    dev: list[Task]
    heldout: list[Task]
    heldout_twist: str
    seed: int
    hardness: int

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "hardness": self.hardness,
            "heldout_twist": self.heldout_twist,
            "train": [t.to_dict() for t in self.train],
            "dev": [t.to_dict() for t in self.dev],
            "heldout": [t.to_dict() for t in self.heldout],
        }

    @classmethod
    def from_dict(cls, d: dict) -> Splits:
        return cls(
            train=[Task.from_dict(t) for t in d["train"]],
            dev=[Task.from_dict(t) for t in d["dev"]],
            heldout=[Task.from_dict(t) for t in d["heldout"]],
            heldout_twist=d["heldout_twist"],
            seed=d["seed"],
            hardness=d["hardness"],
        )


def _tool_names(tasks: list[Task]) -> set[str]:
    return {n for t in tasks for n in (t.tool.name, t.tool.call_name)}


def _twists(tasks: list[Task]) -> set[str]:
    return {t.tool.twist for t in tasks if t.tool.twist is not None}


def assert_no_leak(splits: Splits) -> None:
    """Raise ``LeakError`` if train shares a tool name, domain or twist family with
    held-out, or if dev shares a tool name or domain with train or held-out."""
    train, dev, held = splits.train, splits.dev, splits.heldout
    shared_tools = _tool_names(train) & _tool_names(held)
    if shared_tools:
        raise LeakError(f"held-out tools in train: {sorted(shared_tools)}")
    shared_twists = _twists(train) & (_twists(held) | {splits.heldout_twist})
    if shared_twists:
        raise LeakError(f"held-out twists in train: {sorted(shared_twists)}")
    domains = {name: {t.tool.domain for t in tasks}
               for name, tasks in (("train", train), ("dev", dev), ("heldout", held))}
    for a, b in (("train", "heldout"), ("dev", "train"), ("dev", "heldout")):
        if domains[a] & domains[b]:
            raise LeakError(f"{a} and {b} share domains: {sorted(domains[a] & domains[b])}")
    for a, b in (("dev", train), ("dev", held)):
        if _tool_names(dev) & _tool_names(b):
            raise LeakError("dev shares tool names with train or held-out")


def make_splits(seed: int, hardness: int = 0, tools_per_domain: int = 8) -> Splits:
    rng = random.Random(f"splits:{seed}")
    domains = sorted(DOMAINS)
    rng.shuffle(domains)
    train_d = domains[:_N_TRAIN_DOMAINS]
    held_d = domains[_N_TRAIN_DOMAINS:_N_TRAIN_DOMAINS + _N_HELDOUT_DOMAINS]
    dev_d = domains[_N_TRAIN_DOMAINS + _N_HELDOUT_DOMAINS:]
    held_twist = rng.choice(TWISTS)
    train_twists = [t for t in TWISTS if t != held_twist]
    used: set[str] = set()
    pools = {
        "train": (train_d, [None, *train_twists], TRAIN_SIZE),
        "heldout": (held_d, [held_twist, None, held_twist, None, held_twist], HELDOUT_SIZE),
        "dev": (dev_d, [None, *TWISTS], DEV_SIZE),
    }
    built = {}
    for name, (split_domains, pattern, size) in pools.items():
        tools = generate_tools(random.Random(f"{seed}:{name}:tools"), split_domains, pattern,
                               tools_per_domain, used, hardness)
        built[name] = generate_tasks(tools, size, random.Random(f"{seed}:{name}:tasks"), name)
    splits = Splits(built["train"], built["dev"], built["heldout"], held_twist, seed, hardness)
    assert_no_leak(splits)
    return splits


def splits_to_json(splits: Splits) -> str:
    return json.dumps(splits.to_dict(), indent=1)


def splits_from_json(text: str) -> Splits:
    return Splits.from_dict(json.loads(text))
