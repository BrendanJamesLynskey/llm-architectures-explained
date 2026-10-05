"""Every model file validates against the JSON Schema, every field names a
source that the file lists, and estimates only ever appear labelled."""

from __future__ import annotations

import json

import jsonschema
import pytest

from helpers import ROOT, models, walk_fields

SCHEMA = json.loads((ROOT / "data/schema/model.schema.json").read_text())
IDS = [m["id"] for m in models()]


@pytest.mark.parametrize("m", models(), ids=IDS)
def test_validates(m):
    jsonschema.validate(m, SCHEMA)


@pytest.mark.parametrize("m", models(), ids=IDS)
def test_every_source_is_listed(m):
    for path, f in walk_fields({"facts": m["facts"], "arch": m["arch"]}):
        if f["st"] == "not-disclosed":
            continue
        assert f.get("src") in m["sources"], f"{m['id']}: {path} cites {f.get('src')!r}, not in sources"
    for e in m["facts"].get("estimates", []):
        assert m["sources"][e["src"]]["type"] == "reported-estimate"


@pytest.mark.parametrize("m", models(), ids=IDS)
def test_estimates_never_pose_as_facts(m):
    """A reported estimate lives only in facts.estimates, never in a fact or
    an architecture field, and only for closed models."""
    for path, f in walk_fields({"facts": {k: v for k, v in m["facts"].items() if k != "estimates"}, "arch": m["arch"]}):
        assert f["st"] != "reported-estimate", f"{m['id']}: {path}"
    if m["facts"].get("estimates"):
        assert not m["open_weights"]


def test_ids_unique_and_file_names_match():
    assert len(set(IDS)) == len(IDS)
    for p in (ROOT / "data/models").glob("*.yaml"):
        assert p.stem in IDS


def test_closed_models_without_dimensions_say_why():
    for m in models():
        if m["arch"] is None:
            assert m["no_arch"], m["id"]
