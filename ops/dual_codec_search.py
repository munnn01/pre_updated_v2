#!/usr/bin/env python
"""A/B/C pilot: fit on TRAIN, calibrate on disjoint TRAIN, evaluate on DEV only."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import torch
import torch.nn.functional as F

from ops.codec_search_ar import QPS, as_video, compare, predict_and_feature
from ops.rcts_pilot import balanced_indices, clip_id
from src.codecs.standard import StandardCodec, ffmpeg_available
from src.data.video_dataset import VideoClipDataset
from src.models.codec_search import CANDIDATES, make_candidates, normalized_bpp
from src.models.dual_codec_search import (MODELS, observations, policy_grid, risk_features,
                                          risk_scores, select_observations)
from src.tasks.action_recognition import ActionRecognitionAnalyzer

FIELDS = ("correct", "cross_correct")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def source_id(record):
    # Kinetics files conventionally name the original YouTube ID before times.
    # Group temporal excerpts when this format is present, otherwise use clip ID.
    import re
    stem = Path(record["path"]).stem
    match = re.match(r"^([A-Za-z0-9_-]{11})(?:_\d+(?:_\d+)?|)$", stem)
    return match.group(1) if match else clip_id(record)


def split_plan(index, fit_count, calibration_count, dev_count):
    if min(fit_count, calibration_count, dev_count) < 20:
        raise ValueError("pilot needs at least 20 videos per subset")
    # Exclude sources in VAL/TEST even if the upstream index has cross-split excerpts.
    excluded = {source_id(r) for split in ("val", "test") for r in index[split]}
    seen = set(excluded)
    pool, original = [], []
    for i, row in enumerate(index["train"]):
        key = source_id(row)
        if key not in seen:
            seen.add(key)
            pool.append(row)
            original.append(i)
    fit_local = balanced_indices(pool, fit_count, "dual-v2-fit-20260924")
    used = set(fit_local)
    remaining = [i for i in range(len(pool)) if i not in used]
    calibration_local = balanced_indices([pool[i] for i in remaining], calibration_count,
                                         "dual-v2-calibration-20260924")
    dev_pool, dev_original, dev_seen = [], [], set()
    for i, row in enumerate(index["val"]):
        key = source_id(row)
        if key not in dev_seen:
            dev_pool.append(row)
            dev_original.append(i)
            dev_seen.add(key)
    dev_local = balanced_indices(dev_pool, dev_count, "dual-v2-dev-20260924")
    plan = {"fit": [original[i] for i in fit_local],
            "calibration": [original[remaining[i]] for i in calibration_local],
            "dev": [dev_original[i] for i in dev_local]}
    for stage, count in (("fit", fit_count), ("calibration", calibration_count), ("dev", dev_count)):
        if len(plan[stage]) != count:
            raise ValueError(f"insufficient disjoint sources for {stage}")
    return plan


def signals(logits, feature, source, anchor):
    sp, sf = source
    ap, _ = anchor
    p = logits.softmax(1)
    top = p.topk(2, dim=1).values[0]
    stop = sp.topk(2, dim=1).values[0]
    return {
        "kl_source": float(F.kl_div(logits.log_softmax(1), sp, reduction="batchmean")),
        "feature_distance": float(1 - F.cosine_similarity(feature, sf).item()),
        "source_confidence": float(stop[0]), "source_margin": float(stop[0] - stop[1]),
        "source_top1_agrees": bool(p.argmax(1).item() == sp.argmax(1).item()),
        "confidence": float(top[0]), "margin": float(top[0] - top[1]),
        "entropy": float(-(p * p.clamp_min(1e-12).log()).sum()),
        "source_top1_prob": float(p[0, sp.argmax(1).item()]),
        "kl_anchor": float(F.kl_div(logits.log_softmax(1), ap, reduction="batchmean")),
        "anchor_top1_agrees": bool(p.argmax(1).item() == ap.argmax(1).item()),
    }


def measure(dataset, index, analyzers, codec):
    source, label, meta = dataset[index]
    rgb = (source.permute(1, 2, 3, 0).numpy() * 255).round().astype(np.uint8)
    # The upstream dataset silently substitutes black for failed decode. Reject it.
    import cv2
    cap = cv2.VideoCapture(dataset.samples[index]["path"])
    ok, _ = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"undecodable source: {meta['sequence_id']}")
    variants = make_candidates(rgb)
    clean = {}
    for model, analyzer in analyzers.items():
        logits, feature = predict_and_feature(analyzer, source[None])
        clean[model] = (logits.softmax(1), feature)
    measurements = []
    for qp in QPS:
        candidates, anchor = [], {}
        for name, candidate in variants.items():
            t, h, w, _ = candidate.shape
            reconstructed, native = codec._encode_decode_clip(candidate, qp=qp)
            row = {"name": name, "bpp": normalized_bpp(native, h, w),
                   "coded_bytes": int(round(native * t * h * w / 8)),
                   "encoded_size": [h, w], "signals": {}}
            for model, field in zip(MODELS, FIELDS):
                logits, feature = predict_and_feature(analyzers[model], as_video(reconstructed))
                if name == "identity128":
                    anchor[model] = (logits.softmax(1), feature)
                row["signals"][model] = signals(logits, feature, clean[model], anchor[model])
                row[field] = bool(logits.argmax(1).item() == label)
            candidates.append(row)
        measurements.append({"qp": qp, "candidates": candidates})
    return {"schema": 3, "sequence_id": meta["sequence_id"], "codec": codec.codec,
            "measurements": measurements}


def validate_row(row, expected_id, codec, cache_key):
    if (row.get("schema") != 3 or row.get("sequence_id") != expected_id
            or row.get("codec") != codec or row.get("cache_key") != cache_key):
        raise ValueError("stale or mismatched dual-analyzer cache")
    if [m["qp"] for m in row["measurements"]] != list(QPS):
        raise ValueError("incomplete QP cache")
    for m in row["measurements"]:
        if [r["name"] for r in m["candidates"]] != list(CANDIDATES):
            raise ValueError("incomplete candidate cache")
        observations(m["candidates"])
        for candidate in m["candidates"]:
            if not all(isinstance(candidate[f], bool) for f in FIELDS):
                raise ValueError("missing analyzer outcomes")


def collect(dataset, indices, analyzers, codec, out, stage, cache_key):
    rows = []
    for number, i in enumerate(indices, 1):
        path = out / "cache" / stage / f"clip_{i:05d}.json"
        if path.exists():
            row = json.loads(path.read_text(encoding="utf-8"))
        else:
            row = measure(dataset, i, analyzers, codec)
            row["cache_key"] = cache_key
            write_json(path, row)
        validate_row(row, clip_id(dataset.samples[i]), codec.codec, cache_key)
        rows.append(row)
        print(f"[{stage}] {number}/{len(indices)} codec={codec.codec}", flush=True)
    return rows


def fit_risk(rows):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    x, y = [], []
    for row in rows:
        for m in row["measurements"]:
            obs = observations(m["candidates"])
            for i, candidate in enumerate(m["candidates"][1:], 1):
                if candidate["bpp"] >= obs[0]["bpp"]:
                    continue
                x.append(risk_features(obs, i, m["qp"]))
                y.append([int(m["candidates"][0][f] and not candidate[f]) for f in FIELDS])
    if not x:
        raise ValueError("no bitrate-saving training candidates")
    x, y = np.stack(x), np.asarray(y)
    state = {"schema": 1, "models": {}, "n_fit_videos": len(rows), "n_candidate_rows": len(x),
             "note": "Uncalibrated logistic risk scores; thresholds fitted on disjoint calibration videos."}
    for j, model in enumerate(MODELS):
        if len(np.unique(y[:, j])) < 2:
            # With no positive examples we cannot learn harm: fail conservatively.
            state["models"][model] = {"constant": 1., "reason": "single-class training outcomes"}
            continue
        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=2000, random_state=53))
        pipe.fit(x, y[:, j])
        scaler, estimator = pipe.steps[0][1], pipe.steps[1][1]
        state["models"][model] = {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
            "coef": estimator.coef_[0].tolist(), "intercept": float(estimator.intercept_[0]),
            "positive_count": int(y[:, j].sum())}
    return state


def prepare(rows, state):
    result = []
    for row in rows:
        measurements = []
        for m in row["measurements"]:
            obs = observations(m["candidates"])
            measurements.append((m, obs, risk_scores(obs, m["qp"], state)))
        result.append(measurements)
    return result


def selected_arrays(prepared, policy):
    # [video, QP, rate + two correctness outcomes]. Choices shared by analyzers.
    values, choices = [], Counter()
    for measurements in prepared:
        video = []
        for m, obs, risks in measurements:
            i = select_observations(obs, m["qp"], policy, risks)
            chosen = m["candidates"][i]
            video.append([chosen["bpp"], *[float(chosen[f]) for f in FIELDS]])
            choices[chosen["name"]] += 1
        values.append(video)
    return np.asarray(values), dict(choices)


def metrics(base, trial):
    def curve(array, j):
        mean = array.mean(axis=0)
        return {str(q): {"bpp": float(mean[i, 0]), "top1": float(mean[i, j]), "n": len(array)}
                for i, q in enumerate(QPS)}
    report = {}
    for j, model in enumerate(MODELS, 1):
        a, b = curve(base, j), curve(trial, j)
        score = compare(a, b)
        score["bd_accuracy_top1_pp"] = (100 * score["bd_accuracy_top1"]
                                         if score["bd_accuracy_top1"] is not None else None)
        report[model] = {"metrics": score, "anchor_curve": a, "trial_curve": b}
    return report


def eligible(report):
    return all(r["metrics"]["bd_rate_top1_pct"] is not None
               and r["metrics"]["bd_rate_top1_pct"] < 0
               and r["metrics"]["bd_accuracy_top1"] is not None
               and r["metrics"]["bd_accuracy_top1"] > 0
               and r["metrics"]["min_same_qp_top1_gap"] >= -.01 - 1e-12
               for r in report.values())


def calibrate(prepared):
    base, _ = selected_arrays(prepared, {"mode": "identity"})
    grid, chosen = [], {}
    for policy in policy_grid():
        trial, choices = selected_arrays(prepared, policy)
        scores = metrics(base, trial)
        row = {"policy": policy, "analyzers": scores, "eligible": eligible(scores), "choices": choices}
        row["worst_bd_rate"] = max(r["metrics"]["bd_rate_top1_pct"] for r in scores.values()) if row["eligible"] else None
        grid.append(row)
    for arm in ("A", "B", "C"):
        valid = [r for r in grid if r["policy"]["mode"] == arm and r["eligible"]]
        chosen[arm] = min(valid, key=lambda r: r["worst_bd_rate"])["policy"] if valid else {"mode": "identity"}
    valid = [r for r in grid if r["eligible"]]
    winner = min(valid, key=lambda r: r["worst_bd_rate"])["policy"] if valid else {"mode": "identity"}
    return {"policies": chosen, "selected_policy": winner, "grid": grid}


def bootstrap(base, trial, draws, seed=20260924):
    rng = np.random.default_rng(seed)
    samples = {model: {key: [] for key in ("bd_rate_top1_pct", "bd_accuracy_top1_pp")} for model in MODELS}
    for _ in range(draws):
        idx = rng.integers(0, len(base), len(base))
        report = metrics(base[idx], trial[idx])
        for model in MODELS:
            for key in samples[model]:
                value = report[model]["metrics"][key]
                if value is not None:
                    samples[model][key].append(value)
    return {model: {key: {"valid_draws": len(v), "requested_draws": draws,
                           "ci95": np.percentile(v, [2.5, 97.5]).tolist() if v else None}
                    for key, v in values.items()} for model, values in samples.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--codec", choices=("h264", "h265"), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fit-clips", type=int, default=400)
    parser.add_argument("--calibration-clips", type=int, default=200)
    parser.add_argument("--dev-clips", type=int, default=200)
    parser.add_argument("--bootstrap", type=int, default=500)
    args = parser.parse_args()
    if not ffmpeg_available() or args.bootstrap < 1:
        raise ValueError("ffmpeg/ffprobe and positive bootstrap count required")
    np.random.seed(53)
    torch.manual_seed(53)
    torch.set_num_threads(2)
    index = json.loads(args.index.read_text(encoding="utf-8"))
    plan = split_plan(index, args.fit_clips, args.calibration_clips, args.dev_clips)
    ids = {stage: [clip_id(index["val" if stage == "dev" else "train"][i]) for i in indices]
           for stage, indices in plan.items()}
    import sklearn
    import torchvision
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    manifest = {"experiment": "dual_codec_search_v2_pilot", "codec": args.codec, "qps": list(QPS),
        "candidates": list(CANDIDATES), "models": list(MODELS), "code_commit": commit,
        "frames": 16, "frame_size": 128, "temporal_stride": 2, "preset": "medium",
        "weights": "torchvision KINETICS400_V1, frozen", "split_ids": ids,
        "split_fingerprints": {stage: digest(keys) for stage, keys in ids.items()},
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "torchvision": torchvision.__version__, "sklearn": sklearn.__version__,
                     "ffmpeg": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0]},
        "scope": "Development pilot. Both analyzers are optimization targets, not held-out models.",
        "selection": "Fit risk on TRAIN-fit; tune thresholds only on TRAIN-calibration; never TEST."}
    cache_key = digest(manifest)
    out = args.out_dir
    if (out / "manifest.json").exists() and json.loads((out / "manifest.json").read_text()) != manifest:
        raise ValueError("output directory belongs to a different run; do not mix caches")
    write_json(out / "manifest.json", manifest)
    print(f"[dual-v2] codec={args.codec} counts={ {s: len(v) for s, v in ids.items()} }", flush=True)
    datasets = {split: VideoClipDataset(args.index, split=split, num_frames=16, frame_size=128,
                                        temporal_stride=2, train=False, return_metadata=True)
                for split in ("train", "val")}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    analyzers = {model: ActionRecognitionAnalyzer(model, clip_size=112).freeze().to(device) for model in MODELS}
    codec = StandardCodec(args.codec, preset="medium", strict_decode=True)
    start = time.monotonic()
    fit = collect(datasets["train"], plan["fit"], analyzers, codec, out, "fit", cache_key)
    state = fit_risk(fit)
    write_json(out / "risk_model.json", state)
    calibration = collect(datasets["train"], plan["calibration"], analyzers, codec, out, "calibration", cache_key)
    calibrated = calibrate(prepare(calibration, state))
    write_json(out / "calibration.json", calibrated)
    frozen = {"manifest_sha256": cache_key, "risk_sha256": digest(state),
              "policies": calibrated["policies"], "selected_policy": calibrated["selected_policy"]}
    write_json(out / "frozen_policy.json", frozen)  # freeze BEFORE seeing DEV outcomes
    print(f"[calibrated] {json.dumps(frozen)}", flush=True)
    dev = collect(datasets["val"], plan["dev"], analyzers, codec, out, "dev", cache_key)
    prepared = prepare(dev, state)
    base, _ = selected_arrays(prepared, {"mode": "identity"})
    policies = {"v1": {"mode": "v1"}, **calibrated["policies"], "selected": calibrated["selected_policy"]}
    report = {"manifest": manifest, "frozen": frozen, "arms": {}, "bootstrap_unit": "whole video, all QPs paired"}
    for name, policy in policies.items():
        trial, choices = selected_arrays(prepared, policy)
        scores = metrics(base, trial)
        report["arms"][name] = {"policy": policy, "analyzers": scores, "choices": choices,
            "bootstrap": bootstrap(base, trial, args.bootstrap),
            "point_target_met": all(r["metrics"]["bd_rate_top1_pct"] is not None
                                    and r["metrics"]["bd_rate_top1_pct"] < -15
                                    and r["metrics"]["bd_accuracy_top1"] is not None
                                    and r["metrics"]["bd_accuracy_top1"] > 0 for r in scores.values())}
    report["elapsed_seconds"] = time.monotonic() - start
    write_json(out / "pilot_result.json", report)
    print(json.dumps({"codec": args.codec, "selected": report["arms"]["selected"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
