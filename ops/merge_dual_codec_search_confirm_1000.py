#!/usr/bin/env python
"""Validate two 500-video shards and compute one 1,000-video paired result."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ops.codec_search_ar import QPS
from ops.dual_codec_search import (bootstrap, digest, metrics, prepare,
                                   selected_arrays, write_json)
from ops.dual_codec_search_confirm_1000 import CONFIG, file_sha256, load_frozen
from src.models.codec_search import CANDIDATES
from src.models.dual_codec_search import MODELS, observations


def fingerprint(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()[:16]


def load_shards(directories: list[Path], codec: str, config: dict) -> tuple[list[dict], list[dict]]:
    if len(directories) != 2:
        raise ValueError("exactly two shard directories are required")
    pairs = []
    for directory in directories:
        report = json.loads((directory / "shard_result.json").read_text(encoding="utf-8"))
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if report["manifest"] != manifest:
            raise ValueError("shard result and manifest disagree")
        with (directory / "shard_records.jsonl").open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream]
        if (len(rows) != manifest["n"] or len(rows) != 500
                or [row["sequence_id"] for row in rows] != manifest["sample_ids"]):
            raise ValueError("missing, reordered or mismatched shard records")
        if fingerprint(manifest["sample_ids"]) != manifest["shard_fingerprint"]:
            raise ValueError("shard fingerprint mismatch")
        cache_key = digest(manifest)
        for row in rows:
            if (row.get("schema") != 3 or row.get("codec") != codec
                    or row.get("cache_key") != cache_key
                    or [m["qp"] for m in row.get("measurements", [])] != list(QPS)):
                raise ValueError("invalid or stale per-video record")
            for measurement in row["measurements"]:
                candidates = measurement["candidates"]
                if [candidate["name"] for candidate in candidates] != list(CANDIDATES):
                    raise ValueError("candidate order changed")
                observations(candidates)  # read-only label-free validation
                if any(not isinstance(candidate.get(field), bool)
                       for candidate in candidates for field in ("correct", "cross_correct")):
                    raise ValueError("missing analyzer correctness outcome")
        pairs.append((manifest, rows))
    pairs.sort(key=lambda pair: pair[0]["shard"])
    manifests = [pair[0] for pair in pairs]
    if [m["shard"] for m in manifests] != [0, 1]:
        raise ValueError("missing or duplicate shard")
    common = ("experiment", "codec", "shards", "total_expected", "split",
              "test_fingerprint", "qps", "candidates", "models", "config_sha256",
              "code_commit", "pilot_code_commit", "pilot_manifest_sha256",
              "risk_model_sha256", "frozen_policy_sha256", "selected_policy")
    for key in common:
        if manifests[0][key] != manifests[1][key]:
            raise ValueError(f"shard provenance mismatch: {key}")
    if (manifests[0]["experiment"] != config["experiment"]
            or manifests[0]["codec"] != codec or manifests[0]["split"] != "test"
            or manifests[0]["total_expected"] != config["test_clips"]
            or manifests[0]["test_fingerprint"] != config["test_fingerprint"]
            or manifests[0]["config_sha256"] != file_sha256(CONFIG)
            or manifests[0]["pilot_code_commit"] != config["pilot_code_commit"]
            or manifests[0]["risk_model_sha256"] !=
                config["pilot_artifacts"][codec]["risk_model_sha256"]
            or manifests[0]["frozen_policy_sha256"] !=
                config["pilot_artifacts"][codec]["frozen_policy_sha256"]):
        raise ValueError("not the frozen V2 1,000-video comparison")
    rows = pairs[0][1] + pairs[1][1]
    ids = [row["sequence_id"] for row in rows]
    if (len(ids) != config["test_clips"] or len(set(ids)) != len(ids)
            or fingerprint(ids) != config["test_fingerprint"]):
        raise ValueError("1,000-video fingerprint, count or uniqueness mismatch")
    return rows, manifests


def aggregate(rows: list[dict], manifests: list[dict], codec: str, config: dict) -> dict:
    risk, frozen, selected = load_frozen(codec, config)
    if (manifests[0]["selected_policy"] != selected
            or manifests[0]["pilot_manifest_sha256"] != frozen["manifest_sha256"]):
        raise ValueError("shard selected policy is not the pilot policy")
    prepared = prepare(rows, risk)
    base, _ = selected_arrays(prepared, {"mode": "identity"})
    arms = {}
    for name, policy in (("v1", {"mode": "v1"}), ("dual_v2", selected)):
        trial, choices = selected_arrays(prepared, policy)
        scores = metrics(base, trial)
        intervals = bootstrap(base, trial, config["bootstrap_draws"])
        arms[name] = {"policy": policy, "choices": choices,
                      "analyzers": scores, "bootstrap": intervals}
    v2 = arms["dual_v2"]["analyzers"]
    met = all(v2[model]["metrics"]["bd_rate_top1_pct"] is not None
              and v2[model]["metrics"]["bd_rate_top1_pct"] < -15
              and v2[model]["metrics"]["bd_accuracy_top1"] is not None
              and v2[model]["metrics"]["bd_accuracy_top1"] > 0 for model in MODELS)
    guard = all(v2[model]["metrics"]["min_same_qp_top1_gap"] >= -.01 - 1e-12
                for model in MODELS)
    return {"experiment": config["experiment"], "codec": codec, "n": len(rows),
            "split": "test", "test_fingerprint": config["test_fingerprint"],
            "source_shards": [dict(manifest) for manifest in manifests],
            "arms": arms, "point_target_met": met, "same_qp_guard_met": guard,
            "bootstrap_unit": "whole video with all QPs paired",
            "interpretation": config["interpretation"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codec", choices=("h264", "h265"), required=True)
    parser.add_argument("--shard-dir", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    rows, manifests = load_shards(args.shard_dir, args.codec, config)
    report = aggregate(rows, manifests, args.codec, config)
    write_json(args.out, report)
    print(json.dumps({"codec": args.codec, "n": report["n"],
                      "point_target_met": report["point_target_met"],
                      "same_qp_guard_met": report["same_qp_guard_met"],
                      "analyzers": {model: report["arms"]["dual_v2"]["analyzers"][model]["metrics"]
                                    for model in MODELS}}, indent=2))


if __name__ == "__main__":
    main()
