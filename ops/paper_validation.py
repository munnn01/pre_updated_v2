#!/usr/bin/env python
"""Exploratory, paired 1,000-video reanalysis of frozen V2 candidate records.

The TEST set was previously inspected in V1 work. This script never tunes a
policy or chooses a fixed comparator using TEST outcomes. It reports *all*
prespecified fixed candidates and the pilot-frozen A/B/C policies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np

from ops.dual_codec_search import (
    bootstrap,
    metrics,
    prepare,
    selected_arrays,
    write_json,
)
from ops.dual_codec_search_confirm_1000 import CONFIG, load_frozen
from ops.merge_dual_codec_search_confirm_1000 import load_shards
from src.models.codec_search import CANDIDATES
from src.models.dual_codec_search import MODELS


def fixed_arrays(rows: list[dict], name: str) -> tuple[np.ndarray, dict[str, int]]:
    """One predetermined transform at every QP; never inspect outcomes to choose."""
    if name not in CANDIDATES:
        raise ValueError(f"unknown fixed candidate: {name}")
    index = CANDIDATES.index(name)
    values = []
    for row in rows:
        clip = []
        for measurement in row["measurements"]:
            candidate = measurement["candidates"][index]
            if candidate["name"] != name:
                raise ValueError("candidate order does not match frozen design")
            clip.append([candidate["bpp"], float(candidate["correct"]),
                         float(candidate["cross_correct"])])
        values.append(clip)
    return np.asarray(values, dtype=np.float64), {name: len(rows) * len(rows[0]["measurements"])}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate(rows: list[dict], codec: str, draws: int) -> dict:
    """Evaluate every arm on paired clips; policy parameters are read-only."""
    if draws < 1:
        raise ValueError("bootstrap draws must be positive")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    risk, frozen, selected = load_frozen(codec, config)
    prepared = prepare(rows, risk)
    anchor, anchor_choices = selected_arrays(prepared, {"mode": "identity"})
    arrays: dict[str, np.ndarray] = {"identity128": anchor}
    choices: dict[str, dict[str, int]] = {"identity128": anchor_choices}
    policies: dict[str, dict | None] = {"identity128": {"mode": "identity"}}
    for name in CANDIDATES[1:]:
        arrays[name], choices[name] = fixed_arrays(rows, name)
        policies[name] = None  # a fixed transform, not a learned policy
    for name, policy in (("v1", {"mode": "v1"}),
                         ("A", frozen["policies"]["A"]),
                         ("B", frozen["policies"]["B"]),
                         ("C", frozen["policies"]["C"])):
        arrays[name], choices[name] = selected_arrays(prepared, policy)
        policies[name] = policy
    if selected != frozen["policies"]["C"]:
        raise ValueError("selected V2 policy differs from frozen C")
    arms = {}
    for name, trial in arrays.items():
        if name == "identity128":
            continue
        arms[name] = {"policy": policies[name], "choices": choices[name],
                      "analyzers": metrics(anchor, trial),
                      "bootstrap": bootstrap(anchor, trial, draws)}
    # Direct paired comparison: this is not obtained by subtracting two CIs.
    c_vs_b = {"analyzers": metrics(arrays["B"], arrays["C"]),
              "bootstrap": bootstrap(arrays["B"], arrays["C"], draws)}
    return {"codec": codec, "n": len(rows), "qps": config["qps"],
            "candidates": list(CANDIDATES), "analyzers": list(MODELS),
            "bootstrap_draws": draws,
            "bootstrap_unit": "source video; all QPs and arms remain paired",
            "status": "exploratory post-hoc reanalysis of previously inspected TEST",
            "identity_choices": choices["identity128"], "arms": arms,
            "C_vs_B": c_vs_b}


def markdown(report: dict) -> str:
    def fmt(value: float | None) -> str:
        return "NA" if value is None else f"{value:+.2f}"

    lines = [f"# Paper-validation reanalysis: {report['codec'].upper()}", "",
             f"Paired source videos: **{report['n']}**. This TEST set was previously",
             "inspected; these added comparisons are **exploratory**, not an",
             "independent confirmation. All fixed candidates and frozen A/B/C",
             "arms are reported, without choosing a winner on TEST.", "",
             ("| Arm | Analyzer | BD-rate Top-1 (%) | 95% clip-bootstrap CI | "
              "BD-accuracy (pp) | Minimum same-QP Top-1 gap (pp) |"),
             "|---|---|---:|---:|---:|---:|"]
    for arm, payload in report["arms"].items():
        for model in MODELS:
            met = payload["analyzers"][model]["metrics"]
            ci = payload["bootstrap"][model]["bd_rate_top1_pct"]["ci95"]
            interval = "NA" if ci is None else f"[{ci[0]:+.2f}, {ci[1]:+.2f}]"
            lines.append(f"| `{arm}` | `{model}` | {fmt(met['bd_rate_top1_pct'])} "
                         f"| {interval} | {fmt(met['bd_accuracy_top1_pp'])} "
                         f"| {100 * met['min_same_qp_top1_gap']:+.2f} |")
    lines.extend(["", "## Direct frozen C versus frozen B", "",
                  "This is a paired C-vs-B rate--accuracy comparison, not the",
                  "difference between their BD-rates against identity.", "",
                  "| Analyzer | C vs B BD-rate Top-1 (%) | 95% clip-bootstrap CI |",
                  "|---|---:|---:|"])
    for model in MODELS:
        payload = report["C_vs_B"]
        met = payload["analyzers"][model]["metrics"]
        ci = payload["bootstrap"][model]["bd_rate_top1_pct"]["ci95"]
        interval = "NA" if ci is None else f"[{ci[0]:+.2f}, {ci[1]:+.2f}]"
        lines.append(f"| `{model}` | {fmt(met['bd_rate_top1_pct'])} | {interval} |")
    lines.extend(["", "A negative BD-rate favors the arm named first in the",
                  "header only for C vs B; in the main table it favors the",
                  "listed arm over the identity128 anchor. The full JSON contains",
                  "every QP's curve, choices, provenance and bootstrap counts.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codec", choices=("h264", "h265"), required=True)
    parser.add_argument("--shard-dir", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    rows, manifests = load_shards(args.shard_dir, args.codec, config)
    report = evaluate(rows, args.codec, args.bootstrap)
    report["source"] = [{"shard_directory_name": path.name,
                         "manifest_sha256": file_sha256(path / "manifest.json"),
                         "records_sha256": file_sha256(path / "shard_records.jsonl")}
                        for path in args.shard_dir]
    report["test_fingerprint"] = config["test_fingerprint"]
    report["source_shard_ids"] = [m["shard"] for m in manifests]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / f"{args.codec}_paper_validation.json", report)
    (args.out_dir / f"{args.codec}_paper_validation.md").write_text(
        markdown(report), encoding="utf-8")
    print(f"[{args.codec}] wrote exploratory paired analysis to {args.out_dir}")


if __name__ == "__main__":
    main()
