#!/usr/bin/env python
"""Create and push commit-pinned private Kaggle visual-evaluation notebooks."""
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
from ops.push_rcts_pilot import account_environment, kaggle_command, notebook

TEMPLATES = {"ar": "paper_ar_visual_cell.sh", "coco": "paper_coco_visual_cell.sh"}
DATASETS = {"ar": ["qktttttttttt/kineticscleaned",
                   "qktttttttttt/v2-paper-cache-1000-20260924"],
            "coco": ["awsaf49/coco-2017-dataset"]}


def payload(commit: str, account: str, slug: str, task: str) -> tuple[dict, dict]:
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("full commit SHA required")
    if not re.fullmatch(r"[a-z0-9]+", account):
        raise ValueError("invalid Kaggle account")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", slug):
        raise ValueError("invalid Kaggle slug")
    script = (REPO / "kaggle" / TEMPLATES[task]).read_text(encoding="utf-8")
    book = notebook(script.replace("__REF__", commit), task)
    book["cells"][0]["id"] = f"paper-{task}-visual"
    metadata = {"id": f"{account}/{slug}", "title": slug,
                "code_file": "notebook.ipynb", "language": "python",
                "kernel_type": "notebook", "is_private": True,
                "enable_gpu": task == "coco", "enable_internet": True,
                "dataset_sources": DATASETS[task], "kernel_sources": [],
                "competition_sources": [], "model_sources": []}
    return book, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--account", required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--task", choices=sorted(TEMPLATES), required=True)
    parser.add_argument("--pool", type=Path, default=Path("D:/STUDY/LAB/pool.json"))
    parser.add_argument("--write-only", action="store_true")
    args = parser.parse_args()
    book, meta = payload(args.commit, args.account, args.slug, args.task)
    target = REPO / "ops/_push" / args.account / args.slug
    target.mkdir(parents=True, exist_ok=True)
    (target / "notebook.ipynb").write_text(json.dumps(book), encoding="utf-8")
    (target / "kernel-metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    print(f"[generated] {meta['id']} commit={args.commit}", flush=True)
    if args.write_only:
        return
    env = account_environment(args.pool, args.account)
    env["PYTHONIOENCODING"] = "utf-8"
    require_inactive(meta["id"], env)
    result = subprocess.run(kaggle_command() + ["kernels", "push", "-p", str(target)],
                            capture_output=True, text=True, encoding="utf-8",
                            errors="replace", env=env, check=False)
    output = (result.stdout + result.stderr).strip()
    print(output, flush=True)
    if result.returncode or "successfully pushed" not in output.lower():
        raise SystemExit(result.returncode or 1)


if __name__ == "__main__":
    main()
