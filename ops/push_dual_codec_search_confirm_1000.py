#!/usr/bin/env python
"""Push one commit-pinned private Kaggle notebook for a V2 TEST shard."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ops.push_dual_codec_search import require_inactive
from ops.push_rcts_pilot import account_environment, kaggle_command, metadata, notebook

TEMPLATE = REPO / "kaggle/dual_codec_search_confirm_1000_cell.sh"


def render_cell(commit: str, codec: str, shard: int) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("full commit SHA required")
    if codec not in ("h264", "h265") or shard not in (0, 1):
        raise ValueError("unsupported codec or shard")
    return (TEMPLATE.read_text(encoding="utf-8")
            .replace("__REF__", commit)
            .replace("__CODEC__", codec)
            .replace("__SHARD__", str(shard)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True)
    parser.add_argument("--account", required=True)
    parser.add_argument("--codec", choices=("h264", "h265"), required=True)
    parser.add_argument("--shard", type=int, choices=(0, 1), required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--pool", type=Path, default=Path("D:/STUDY/LAB/pool.json"))
    parser.add_argument("--write-only", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9]+", args.account):
        raise ValueError("invalid account")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", args.slug):
        raise ValueError("invalid notebook slug")
    cell = render_cell(args.commit, args.codec, args.shard)
    payload = notebook(cell, args.codec)
    payload["cells"][0]["id"] = f"dual-v2-confirm-{args.codec}-{args.shard}"
    target = REPO / "ops/_push" / args.account / args.slug
    target.mkdir(parents=True, exist_ok=True)
    (target / "notebook.ipynb").write_text(json.dumps(payload), encoding="utf-8")
    (target / "kernel-metadata.json").write_text(
        json.dumps(metadata(args.account, args.slug)), encoding="utf-8")
    handle = f"{args.account}/{args.slug}"
    print(f"[generated] {handle} codec={args.codec} shard={args.shard} commit={args.commit}", flush=True)
    if args.write_only:
        return
    environment = account_environment(args.pool, args.account)
    environment["PYTHONIOENCODING"] = "utf-8"
    require_inactive(handle, environment)
    response = subprocess.run(kaggle_command() + ["kernels", "push", "-p", str(target)],
                              capture_output=True, text=True, encoding="utf-8",
                              errors="replace", env=environment)
    output = (response.stdout + response.stderr).strip()
    if output:
        print(output, flush=True)
    if response.returncode or "successfully pushed" not in output.lower():
        raise SystemExit(response.returncode or 1)


if __name__ == "__main__":
    main()
