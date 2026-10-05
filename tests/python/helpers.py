from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=None)
def models() -> tuple[dict[str, Any], ...]:
    return tuple(yaml.safe_load(p.read_text()) for p in sorted((ROOT / "data/models").glob("*.yaml")))


def by_id(i: str) -> dict[str, Any]:
    for m in models():
        if m["id"] == i:
            return m
    raise KeyError(i)


def strip(x: Any) -> Any:
    if isinstance(x, dict):
        if "v" in x and "st" in x:
            return strip(x["v"])
        return {k: strip(v) for k, v in x.items()}
    if isinstance(x, list):
        return [strip(i) for i in x]
    return x


def spec(m: dict[str, Any]) -> dict[str, Any]:
    s = strip(m["arch"])
    s["kind"] = m["kind"]
    return s


def walk_fields(tree: Any, path: str = ""):
    if isinstance(tree, dict):
        if "v" in tree and "st" in tree:
            yield path, tree
            return
        for k, v in tree.items():
            yield from walk_fields(v, f"{path}.{k}" if path else k)
    elif isinstance(tree, list):
        for i, v in enumerate(tree):
            yield from walk_fields(v, f"{path}[{i}]")


def snapshot(m: dict[str, Any]) -> dict[str, Any] | None:
    hf = m["sources"].get("hf")
    if not hf:
        return None
    return json.loads((ROOT / hf["snapshot"]).read_text())
