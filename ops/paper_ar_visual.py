#!/usr/bin/env python
"""Preselected paired Kinetics frames for the frozen V2 selector.

The figure is qualitative. Numerical Top-1/BD-rate results come from the
complete paired TEST records, never from the few displayed clips.
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
from PIL import Image, ImageDraw

from ops.dual_codec_search_confirm_1000 import CONFIG, load_frozen, sample_plan
from ops.merge_dual_codec_search_confirm_1000 import load_shards
from src.codecs.standard import StandardCodec, ffmpeg_available
from src.models.codec_search import make_candidates, normalized_bpp
from src.models.dual_codec_search import select


def pick_ids(ids: list[str], count: int, salt: str) -> list[str]:
    """Choose by ID only, without labels, model correctness, or visual quality."""
    if count < 1 or count > len(ids) or len(set(ids)) != len(ids):
        raise ValueError("count must be positive and IDs must be unique")
    return sorted(ids, key=lambda value: (hashlib.sha256(
        f"{salt}|{value}".encode()).hexdigest(), value))[:count]


def _tile(frame: np.ndarray, heading: str, *, side: int = 256) -> Image.Image:
    image = Image.fromarray(frame, "RGB")
    # Nearest-neighbor enlargement is for display only; encoded pixels are
    # retained at their original 96/112/128-pixel spatial dimensions.
    image = image.resize((side, side), Image.Resampling.NEAREST)
    tile = Image.new("RGB", (side, side + 30), "white")
    tile.paste(image, (0, 30))
    ImageDraw.Draw(tile).text((5, 7), heading, fill="black")
    return tile


def save_panel(source: np.ndarray, anchor: np.ndarray, trial: np.ndarray,
               frame_positions: tuple[int, ...], path: Path) -> None:
    side, head = 256, 30
    canvas = Image.new("RGB", (side * 3, (side + head) * len(frame_positions)), "white")
    for row, frame in enumerate(frame_positions):
        for col, (array, title) in enumerate((
            (source, f"Source frame {frame} | 128x128"),
            (anchor, f"Codec anchor | {anchor.shape[1]}x{anchor.shape[2]}"),
            (trial, f"V2 selected | {trial.shape[1]}x{trial.shape[2]}"),
        )):
            canvas.paste(_tile(array[frame], title, side=side),
                         (col * side, row * (side + head)))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def run(args: argparse.Namespace) -> dict:
    if not ffmpeg_available():
        raise ValueError("ffmpeg and ffprobe are required")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if args.qp not in config["qps"]:
        raise ValueError("QP is not on the frozen experiment grid")
    cache, _ = load_shards(args.cache_dir, args.codec, config)
    by_id = {row["sequence_id"]: row for row in cache}
    if len(by_id) != config["test_clips"]:
        raise ValueError("incomplete paired cache")
    risk, _frozen, policy = load_frozen(args.codec, config)
    dataset, indices = sample_plan(args.index, config)
    id_to_index = {dataset.samples[i].get("sequence_id") or
                   "/".join(dataset.samples[i]["path"].replace("\\", "/").split("/")[-2:]): i
                   for i in indices}
    if set(id_to_index) != set(by_id):
        raise ValueError("index and cache video IDs disagree")
    chosen_ids = pick_ids(list(by_id), args.count, "paper-ar-visual-20260924")
    frame_positions = (4, 8, 12)
    codec = StandardCodec(args.codec, preset=config["preset"], strict_decode=True)
    rows = []
    for sequence_id in chosen_ids:
        source, label, meta = dataset[id_to_index[sequence_id]]
        if meta["sequence_id"] != sequence_id:
            raise ValueError("video identity changed during decoding")
        rgb = (source.permute(1, 2, 3, 0).numpy() * 255).round().astype(np.uint8)
        variants = make_candidates(rgb)
        measurement = next(m for m in by_id[sequence_id]["measurements"]
                           if m["qp"] == args.qp)
        position = select(measurement["candidates"], args.qp, policy, risk)
        selected = measurement["candidates"][position]
        anchor, anchor_rate = codec._encode_decode_clip(variants["identity128"], qp=args.qp)
        if selected["name"] == "identity128":
            trial, trial_rate = anchor, anchor_rate
        else:
            trial, trial_rate = codec._encode_decode_clip(variants[selected["name"]], qp=args.qp)
        trial_rate = normalized_bpp(trial_rate, trial.shape[1], trial.shape[2])
        output = args.out_dir / f"{args.codec}_{hashlib.sha256(sequence_id.encode()).hexdigest()[:12]}.png"
        save_panel(rgb, anchor, trial, frame_positions, output)
        rows.append({"sequence_id": sequence_id, "label": label,
                     "codec": args.codec, "qp": args.qp, "chosen": selected["name"],
                     "anchor_bpp_reencoded": anchor_rate,
                     "trial_bpp_reencoded": trial_rate,
                     "anchor_bpp_cached": measurement["candidates"][0]["bpp"],
                     "trial_bpp_cached": selected["bpp"],
                     "r2plus1d_anchor_correct": bool(measurement["candidates"][0]["correct"]),
                     "r2plus1d_trial_correct": bool(selected["correct"]),
                     "r3d_anchor_correct": bool(measurement["candidates"][0]["cross_correct"]),
                     "r3d_trial_correct": bool(selected["cross_correct"]),
                     "panel": output.name})
        print(f"[ar-visual] {args.codec} {sequence_id} -> {selected['name']}", flush=True)
    report = {"purpose": "qualitative paired illustration only",
              "selection": "SHA-256(sequence_id, fixed salt), preselected without metrics",
              "sample_scope": "previously inspected 1000-video TEST set, not new holdout",
              "display": "nearest-neighbor enlargement to 256px panels; no encoded pixels altered",
              "frame_indices_zero_based": frame_positions, "rows": rows}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"{args.codec}_visual_manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, action="append", required=True)
    parser.add_argument("--codec", choices=("h264", "h265"), required=True)
    parser.add_argument("--qp", type=int, default=40)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
