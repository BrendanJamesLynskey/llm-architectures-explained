"""Build data/models/<id>.yaml from the pinned snapshots and the curation.

    python scripts/build_models.py          # write the files
    python scripts/build_models.py --check  # fail if any file would change

Inputs (all committed):
  data/hf/*.json            config.json + API metadata at a pinned revision
  data/transcribed/*.json   configs transcribed from a lab's GitHub or paper
  data/curation/*.yaml      what the config cannot say: names, labs, the
                            totals the authors give, papers, estimates

The output files are generated, but committed, so that the site, the
reference model and the reader all see the same data. CI runs --check.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arch_facts import norm_facts, position_facts  # noqa: E402
from hf_adapter import Cfg, F, adapt, code, model_type_of  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

STATUS = {"card": "disclosed", "name": "disclosed", "blog": "disclosed", "doc": "disclosed",
          "paper": "paper", "config": "config", "code": "code"}


def load_curation() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    g = yaml.safe_load((DATA / "curation" / "gallery.yaml").read_text())
    x = yaml.safe_load((DATA / "curation" / "extras.yaml").read_text())
    for e in g:
        e["_in_gallery"] = True
    for e in x["models"]:
        e["_in_gallery"] = False
    return g + x["models"], x["estimate_sources"]


def snap_path(repo: str) -> Path:
    return DATA / "hf" / (repo.replace("/", "__") + ".json")


def num(v: Any) -> Any:
    """PyYAML (YAML 1.1) reads `1.5e9` as a string; the curation writes
    numbers that way, so convert them back."""
    if isinstance(v, str) and re.fullmatch(r"[0-9.]+e[0-9]+", v):
        f = float(v)
        return int(f) if f.is_integer() else f
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v


def fact(entry: dict[str, Any], key: str) -> dict[str, Any] | None:
    v = entry.get(key)
    if v is None:
        return None
    val, src, where = v
    val = num(val)
    return F(val, STATUS[src], src if src not in ("name",) else "card", where)


def retarget(tree: Any, src: str, st: str, refs: dict[str, str], fname: str) -> Any:
    """Point every config-sourced field of a transcribed config at its real
    source (the lab's GitHub, a paper or a card)."""
    if isinstance(tree, dict):
        if "v" in tree and "st" in tree:
            if tree["st"] == "config":
                key = str(tree.get("ref", "")).split(",")[0].strip()
                out = dict(tree)
                out["st"] = st
                out["src"] = src
                out["ref"] = refs.get(key, f"{fname}: {tree.get('ref')}")
                return out
            return tree
        return {k: retarget(v, src, st, refs, fname) for k, v in tree.items()}
    if isinstance(tree, list):
        return [retarget(i, src, st, refs, fname) for i in tree]
    return tree


def rename_src(tree: Any, old: str, new: str) -> Any:
    if isinstance(tree, dict):
        if "v" in tree and "st" in tree:
            return {**tree, "src": new} if tree.get("src") == old else tree
        return {k: rename_src(v, old, new) for k, v in tree.items()}
    if isinstance(tree, list):
        return [rename_src(i, old, new) for i in tree]
    return tree


def strip(x: Any) -> Any:
    if isinstance(x, dict):
        if "v" in x and "st" in x:
            return strip(x["v"])
        return {k: strip(v) for k, v in x.items()}
    if isinstance(x, list):
        return [strip(i) for i in x]
    return x


def build_one(e: dict[str, Any], est_sources: dict[str, Any], built: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": e["id"],
        "name": e["name"],
        "gallery_name": e.get("gallery"),
        "lab": e["lab"],
        "family": e["family"],
        "open_weights": bool(e.get("open", True)),
        "kind": e.get("kind", "decoder"),
        "multimodal": bool(e.get("multimodal", False)),
    }
    sources: dict[str, Any] = {}
    facts: dict[str, Any] = {}
    snap = None
    cfg = None
    mt = None
    if e.get("hf"):
        snap = json.loads(snap_path(e["hf"]).read_text())
        rev = snap["revision"]
        sources["hf"] = {"type": "config", "repo": snap["repo"], "revision": rev,
                         "url": (f"https://huggingface.co/{snap['repo']}/blob/{rev}/{snap['file']}" if snap["config"] is not None
                                 else f"https://huggingface.co/{snap['repo']}/tree/{rev}"),
                         "snapshot": f"data/hf/{snap_path(e['hf']).name}", "config_available": snap["config"] is not None}
        sources["card"] = {"type": "card", "url": f"https://huggingface.co/{snap['repo']}/blob/{rev}/README.md"}
        cfg = snap["config"]
    if e.get("paper"):
        sources["paper"] = {"type": "paper", "arxiv": e["paper"], "url": f"https://arxiv.org/abs/{e['paper']}"}
    if e.get("blog"):
        sources["blog"] = {"type": "blog", "url": e["blog"]}
    if e.get("doc"):
        sources["doc"] = {"type": "doc", "url": e["doc"]}

    arch = None
    if e.get("tx"):
        tx = json.loads((DATA / "transcribed" / f"{e['tx']}.json").read_text())
        s = tx["_source"]
        sources[s["id"]] = {"type": s["type"], "url": s["url"], "title": s["title"], "transcription": f"data/transcribed/{e['tx']}.json"}
        raw = {k: v for k, v in tx.items() if not k.startswith("_")}
        mt = model_type_of(raw)
        arch = retarget(adapt(raw), s["id"], tx["_status"], tx["_refs"], f"data/transcribed/{e['tx']}.json")
        cfg = raw
    elif e.get("arch_from"):
        base = built[e["arch_from"]]
        arch = rename_src(copy.deepcopy(base["arch"]), "hf", "base_hf")
        mt = base["_model_type"]
        cfg = base["_cfg"]
        out["arch_from"] = {"id": e["arch_from"], "ref": e.get("arch_from_ref")}
        for k, v in base["sources"].items():
            if k in ("hf", "gemma-github", "meta-github"):
                sources["base_" + k] = v
    elif cfg is not None and not e.get("no_arch"):
        mt = model_type_of(cfg)
        arch = adapt(cfg)
    if arch is not None:
        if "kind" in arch:
            out["kind"] = strip(arch.pop("kind"))
        for k in ("mtp_layers", "mtp_layer"):
            if k in arch and e.get("mtp") is None:
                pass
        if e.get("mtp"):
            arch["mtp_layers"] = fact(e, "mtp")
        if e.get("loops"):
            v, src, where = e["loops"]
            arch["loops"] = F(v, STATUS[src], "hf" if src == "config" else src, where)
        if e.get("ced_encoder_layers"):
            arch["ced_encoder_layers"] = fact(e, "ced_encoder_layers")
            arch["ced_window"] = fact(e, "ced_window")
            out["kind"] = "ced"
    out["no_arch"] = e.get("no_arch")
    if arch is not None:
        sources["code"] = {"type": "code", "url": "https://github.com/huggingface/transformers/tree/v5.18.0/src/transformers/models",
                           "title": "transformers 5.18.0 modelling code, or the model repository's own modelling file at the pinned revision"}

    # --- facts --------------------------------------------------------------
    if e.get("released"):
        facts["released"] = fact(e, "released")
    elif snap and snap["api"].get("createdAt"):
        facts["released"] = F(snap["api"]["createdAt"][:7], "config", "hf", "Hugging Face repository creation date (api.createdAt)")
    else:
        facts["released"] = F(None, "not-disclosed")
    lic = snap["api"].get("license") if snap else None
    if lic:
        facts["licence"] = F(lic, "config", "hf", "README metadata: license")
    elif not out["open_weights"]:
        src = next(k for k in ("doc", "blog", "paper") if k in sources)
        facts["licence"] = F("proprietary", "disclosed", src, "weights not published")
    else:
        facts["licence"] = F(None, "not-disclosed")
    for k, name in (("total", "total_params"), ("active", "active_params")):
        facts[name] = fact(e, k) or F(None, "not-disclosed")
    if e.get("context"):
        facts["context"] = fact(e, "context")
    else:
        c = Cfg(cfg) if cfg else None
        key = None
        if c is not None:
            for k in ("max_position_embeddings", "n_positions", "max_seq_len", "seq_length", "context_length"):
                if c.has(k):
                    key = k
                    break
        if key and c is not None:
            f = c.f(key)
            if arch is not None and e.get("tx"):
                tx = json.loads((DATA / "transcribed" / f"{e['tx']}.json").read_text())
                f = F(f["v"], tx["_status"], tx["_source"]["id"], tx["_refs"].get(key, key))
            facts["context"] = f
        else:
            facts["context"] = F(None, "not-disclosed")
    for k in ("moe", "experts"):
        if e.get(k):
            facts[k] = fact(e, k)
    if arch is not None and mt is not None:
        is_mla = "mla" in arch.get("mixers", {})
        pre = Cfg(cfg).prefix if cfg else ""
        nf = norm_facts("transformer" if e["id"] == "transformer-base" else mt, Cfg(cfg).c if cfg else None, is_mla, pre)
        if e["id"] == "transformer-base":
            pos = {"position": F({"scheme": "sinusoidal", "rope_fraction": None, "nope": None}, "paper", "paper", "§3.5: sinusoidal positional encodings")}
        elif e["id"] in ("gpt-3",):
            pos = {"position": F({"scheme": "learned", "rope_fraction": None, "nope": None}, "paper", "paper", "§2.1: same architecture as GPT-2 (learned positions)")}
        elif e["id"] in ("glam", "chinchilla"):
            pos = {"position": F({"scheme": "relative-bias", "rope_fraction": None, "nope": None}, "paper", "paper",
                                 "GLaM §4: per-layer relative positional bias" if e["id"] == "glam" else "Gopher paper §3: relative positional encoding")}
        elif e["id"] == "tiny-aya":
            pos = {"position": F({"scheme": "rope", "rope_fraction": 1.0, "nope": "full-attention layers have no RoPE; sliding-window layers use it"}, "paper", "paper", "§3.1: RoPE on sliding-window layers, NoPE on full-attention layers")}
        elif e["id"] == "palm-540b":
            pos = {"position": F({"scheme": "rope", "rope_fraction": 1.0, "nope": None}, "paper", "paper", "§2: RoPE embeddings")}
        else:
            pos = position_facts(mt, Cfg(cfg).c if cfg else None, arch, pre)
        facts.update(nf)
        facts.update(pos)
        if nf["norm_placement"]["v"] is not None:
            facts["parallel_block"] = code(nf["norm_placement"]["v"] == "parallel", nf["norm_placement"].get("ref", ""))
    if e.get("arch_from"):
        facts = rename_src(facts, "hf", "base_hf") if "hf" not in sources or not sources["hf"]["config_available"] else facts
        for k in ("norm_placement", "norm_type", "qk_norm", "position", "parallel_block"):
            if k in facts:
                facts[k] = rename_src(facts[k], "hf", "base_hf")
        if not out["open_weights"] and not e.get("context"):
            facts["context"] = F(None, "not-disclosed")
    for est in e.get("estimates") or []:
        sid = est["source"]
        s = est_sources[sid]
        sources[sid] = {"type": "reported-estimate", "url": s["url"], "publisher": s["publisher"], "date": s["date"],
                        "where": s["where"], "confidence": " ".join(s["confidence"].split())}
        facts.setdefault("estimates", []).append({"field": est["field"], "v": num(est["value"]), "st": "reported-estimate", "src": sid,
                                                  "quote": est["quote"]})
    out["sources"] = sources
    out["facts"] = facts
    out["arch"] = arch
    checks: dict[str, Any] = {}
    if snap and snap["api"].get("safetensors"):
        stt = snap["api"]["safetensors"]
        dts = sorted((stt.get("parameters") or {}).keys())
        checks["hf_weight_count"] = {"v": stt.get("total"), "dtypes": dts,
                                     "packed": any(d in ("U8", "I8", "I32", "I64") for d in dts),
                                     "ref": "Hugging Face API safetensors.total at the pinned revision"}
    out["checks"] = checks
    out["_model_type"] = mt
    out["_cfg"] = cfg
    return out


def dump(m: dict[str, Any]) -> str:
    m = {k: v for k, v in m.items() if not k.startswith("_")}
    return "# Generated by scripts/build_models.py from data/hf, data/transcribed and data/curation. Do not edit.\n" + yaml.safe_dump(
        m, sort_keys=False, allow_unicode=True, width=120)


def bundles(models: list[dict[str, Any]]) -> dict[str, str]:
    """The site's data: every model (server side) and a compact spec list
    for the client-side compare tool."""
    full = [{k: v for k, v in m.items() if not k.startswith("_")} for m in models]
    compact = []
    for m in full:
        f = m["facts"]
        compact.append({
            "id": m["id"], "name": m["name"], "lab": m["lab"], "open": m["open_weights"], "kind": m["kind"],
            "released": f["released"]["v"],
            "total": {"v": f["total_params"]["v"], "st": f["total_params"]["st"]},
            "active": {"v": f["active_params"]["v"], "st": f["active_params"]["st"]},
            "context": f["context"]["v"],
            "norm": (f.get("norm_placement") or {}).get("v"),
            "estimates": [{"field": e["field"], "v": e["v"], "src": e["src"],
                           "publisher": m["sources"][e["src"]]["publisher"], "date": m["sources"][e["src"]]["date"],
                           "confidence": m["sources"][e["src"]]["confidence"], "url": m["sources"][e["src"]]["url"]}
                          for e in f.get("estimates", [])],
            "spec": ({**strip(m["arch"]), "kind": m["kind"]} if m["arch"] is not None else None),
        })
    dumps = lambda x: json.dumps(x, separators=(",", ":"), ensure_ascii=False) + "\n"  # noqa: E731
    return {"src/data/models.json": dumps(full), "public/data/specs.json": dumps(compact)}


def build_all() -> dict[str, str]:
    entries, est = load_curation()
    built: dict[str, dict[str, Any]] = {}
    pending = list(entries)
    while pending:
        nxt = []
        for e in pending:
            if e.get("arch_from") and e["arch_from"] not in built:
                nxt.append(e)
                continue
            built[e["id"]] = build_one(e, est, built)
        if len(nxt) == len(pending):
            raise SystemExit(f"unresolved arch_from: {[e['id'] for e in nxt]}")
        pending = nxt
    ids = [e["id"] for e in entries]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate ids")
    for e in entries:
        built[e["id"]]["in_gallery"] = e["_in_gallery"]
    out = {f"data/models/{i}.yaml": dump(built[i]) for i in ids}
    out.update(bundles([built[i] for i in sorted(ids)]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    files = build_all()
    (DATA / "models").mkdir(exist_ok=True)
    existing = {f"data/models/{p.name}": p for p in (DATA / "models").glob("*.yaml")}
    if a.check:
        bad = [k for k, t in files.items() if not (ROOT / k).exists() or (ROOT / k).read_text() != t]
        bad += [k for k in existing if k not in files]
        if bad:
            print("out of date:", ", ".join(sorted(bad)))
            return 1
        print(f"{len(files)} generated files up to date")
        return 0
    for k, p in existing.items():
        if k not in files:
            p.unlink()
    for k, t in files.items():
        (ROOT / k).parent.mkdir(parents=True, exist_ok=True)
        (ROOT / k).write_text(t)
    print(f"wrote {len(files)} generated files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
