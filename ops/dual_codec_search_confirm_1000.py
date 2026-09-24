#!/usr/bin/env python
"""One shard of the frozen V2 policy on the previously inspected 1,000 TEST clips."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import torch

from ops.codec_search_ar import QPS
from ops.codec_search_confirm import clip_keys
from ops.dual_codec_search import (collect, digest, metrics, prepare,
                                   selected_arrays, write_json)
from ops.rcts_pilot import balanced_indices, clip_id, fingerprint
from src.codecs.standard import StandardCodec, ffmpeg_available
from src.data.video_dataset import VideoClipDataset
from src.models.codec_search import CANDIDATES
from src.models.dual_codec_search import MODELS
from src.tasks.action_recognition import ActionRecognitionAnalyzer

CONFIG = REPO / "configs/dual_codec_search_v2_confirm_1000.json"
ARTIFACTS = REPO / "configs/dual_codec_search_v2_frozen"


def file_sha256(path: Path) -> str:
    """Hash Git text content with canonical LF on Windows and Kaggle Linux."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load_frozen(codec: str, config: dict) -> tuple[dict, dict, dict]:
    source = config["pilot_artifacts"][codec]
    folder = ARTIFACTS / codec
    risk_path, frozen_path = folder / "risk_model.json", folder / "frozen_policy.json"
    for path, key in ((risk_path, "risk_model_sha256"),
                      (frozen_path, "frozen_policy_sha256")):
        if file_sha256(path) != source[key]:
            raise ValueError(f"frozen pilot artifact SHA-256 mismatch: {path}")
    risk = json.loads(risk_path.read_text(encoding="utf-8"))
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen["risk_sha256"] != digest(risk) or risk["schema"] != 1:
        raise ValueError("frozen risk state does not match policy")
    policy = frozen["selected_policy"]
    if (policy != frozen["policies"]["C"] or policy["mode"] != "C"
            or set(risk["models"]) != set(MODELS)):
        raise ValueError("pilot winner or analyzer list is not the frozen C policy")
    return risk, frozen, policy


def sample_plan(index_path: Path, config: dict) -> tuple[VideoClipDataset, list[int]]:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    keys = clip_keys(index)
    if (keys["train"] & keys["test"]) or (keys["val"] & keys["test"]):
        raise ValueError("TEST clip IDs overlap TRAIN or VAL")
    previous = json.loads((REPO / config["sample_config"]).read_text(encoding="utf-8"))
    if (previous["experiment"] != "codec_search_ar_confirm_v1"
            or previous["confirm_clips"] != config["test_clips"]
            or previous["qps"] != config["qps"]
            or previous["candidates"] != config["candidates"]):
        raise ValueError("V1 paired sample specification changed")
    dataset = VideoClipDataset(index_path, split="test", num_frames=config["frames"],
                               frame_size=config["frame_size"],
                               temporal_stride=config["temporal_stride"],
                               train=False, return_metadata=True)
    indices = balanced_indices(dataset.samples, config["test_clips"], previous["confirm_salt"])
    if (len(indices) != config["test_clips"]
            or len({clip_id(dataset.samples[i]) for i in indices}) != len(indices)
            or fingerprint(dataset.samples, indices) != config["test_fingerprint"]):
        raise ValueError("not the frozen 1,000-video paired TEST sample")
    return dataset, indices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--codec", choices=("h264", "h265"), required=True)
    parser.add_argument("--shard", type=int, choices=(0, 1), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if not ffmpeg_available():
        raise SystemExit("ffmpeg/ffprobe are required")
    random.seed(53)
    np.random.seed(53)
    torch.manual_seed(53)
    torch.set_num_threads(2)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if (config["experiment"] != "dual_codec_search_v2_confirm_1000"
            or tuple(config["qps"]) != QPS
            or tuple(config["candidates"]) != CANDIDATES
            or tuple(config["models"]) != MODELS):
        raise ValueError("frozen configuration disagrees with code")
    risk, frozen, policy = load_frozen(args.codec, config)
    dataset, all_indices = sample_plan(args.index, config)
    indices = [i for position, i in enumerate(all_indices) if position % 2 == args.shard]
    if len(indices) != 500:
        raise ValueError("expected exactly 500 videos per shard")
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    manifest = {
        "experiment": config["experiment"], "codec": args.codec,
        "shard": args.shard, "shards": 2, "n": len(indices),
        "total_expected": config["test_clips"], "split": "test",
        "test_fingerprint": config["test_fingerprint"],
        "shard_fingerprint": fingerprint(dataset.samples, indices),
        "sample_ids": [clip_id(dataset.samples[i]) for i in indices],
        "qps": list(QPS), "candidates": list(CANDIDATES), "models": list(MODELS),
        "config_sha256": file_sha256(CONFIG), "code_commit": git_sha,
        "pilot_code_commit": config["pilot_code_commit"],
        "pilot_manifest_sha256": frozen["manifest_sha256"],
        "risk_model_sha256": file_sha256(ARTIFACTS / args.codec / "risk_model.json"),
        "frozen_policy_sha256": file_sha256(ARTIFACTS / args.codec / "frozen_policy.json"),
        "selected_policy": policy,
        "versions": {"python": platform.python_version(), "torch": torch.__version__},
        "interpretation": config["scope"],
    }
    out = args.out_dir
    if (out / "manifest.json").exists():
        prior = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        if prior != manifest:
            raise ValueError("output cache belongs to a different run")
    write_json(out / "manifest.json", manifest)
    write_json(out / "frozen_policy.json", frozen)
    write_json(out / "risk_model.json", risk)
    print(f"[v2-confirm] codec={args.codec} shard={args.shard} clips=500 "
          f"fingerprint={config['test_fingerprint']} commit={git_sha}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    analyzers = {model: ActionRecognitionAnalyzer(model, clip_size=112).freeze().to(device)
                 for model in MODELS}
    codec = StandardCodec(args.codec, preset=config["preset"], strict_decode=True)
    rows = collect(dataset, indices, analyzers, codec, out, "test", digest(manifest))
    prepared = prepare(rows, risk)
    base, _ = selected_arrays(prepared, {"mode": "identity"})
    arms = {}
    for name, arm_policy in (("v1", {"mode": "v1"}), ("dual_v2", policy)):
        trial, choices = selected_arrays(prepared, arm_policy)
        arms[name] = {"policy": arm_policy, "choices": choices,
                      "analyzers": metrics(base, trial)}
    report = {"manifest": manifest, "arms": arms,
              "warning": "Shard BD-rates are diagnostic only; merge records before final metrics."}
    records = out / "shard_records.jsonl"
    temporary = records.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    temporary.replace(records)
    write_json(out / "shard_result.json", report)
    print(f"[done] codec={args.codec} shard={args.shard} records={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
