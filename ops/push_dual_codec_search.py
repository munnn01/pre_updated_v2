#!/usr/bin/env python
"""Generate/push one private pinned pilot; refuse active duplicates."""
from __future__ import annotations
import argparse
import csv
import io
import json
from pathlib import Path
import re
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from ops.push_rcts_pilot import account_environment, kaggle_command, metadata, notebook


def render_cell(commit, codec):
    if not re.fullmatch(r"[a-f0-9]{40}", commit) or codec not in ("h264", "h265"):
        raise ValueError("a full commit SHA and supported codec are required")
    return ((REPO / "kaggle/dual_codec_search_cell.sh").read_text(encoding="utf-8")
            .replace("__REF__", commit).replace("__CODEC__", codec))


def require_inactive(handle, environment):
    environment = {**environment, "PYTHONIOENCODING": "utf-8"}
    response = subprocess.run(kaggle_command() + ["kernels", "status", handle],
                              capture_output=True, text=True, encoding="utf-8", errors="replace", env=environment)
    output = (response.stdout + response.stderr).lower()
    if any(word in output for word in ("running", "queued", "pending")):
        raise RuntimeError(f"refusing duplicate active run: {handle}")
    if response.returncode and "permission 'kernels.get' was denied" in output:
        # Kaggle returns the same denial for nonexistent and inaccessible slugs.
        # Establish authenticated ownership and absence using the private mine list.
        owner = handle.split("/")[0]
        owned = set()
        for page in range(1, 101):
            listing = subprocess.run(kaggle_command() + ["kernels", "list", "--mine", "--csv",
                                     "--page-size", "100", "--page", str(page)],
                                     capture_output=True, text=True, encoding="utf-8", errors="replace", env=environment)
            if listing.returncode:
                raise RuntimeError("cannot verify authenticated notebook ownership")
            reader = csv.DictReader(io.StringIO(listing.stdout))
            if not reader.fieldnames or "ref" not in reader.fieldnames:
                raise RuntimeError("unexpected authenticated notebook listing")
            refs = [r["ref"] for r in reader]
            if any(not ref.startswith(owner + "/") for ref in refs):
                raise RuntimeError("token belongs to a different account")
            owned.update(refs)
            if handle in owned:
                raise RuntimeError("notebook exists but status unavailable; refusing push")
            if len(refs) < 100:
                if not owned:
                    raise RuntimeError("cannot establish account identity from empty listing")
                return
        raise RuntimeError("notebook enumeration incomplete; refusing push")
    if response.returncode and not any(word in output for word in ("404", "not found", "does not exist")):
        raise RuntimeError(f"cannot establish status of {handle}; check authentication/connectivity")
    if not any(word in output for word in ("complete", "error", "cancel", "404", "not found", "does not exist")):
        raise RuntimeError(f"unknown remote status for {handle}; refusing push")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True)
    parser.add_argument("--account", required=True)
    parser.add_argument("--codec", choices=("h264", "h265"), required=True)
    parser.add_argument("--pool", type=Path, default=Path("D:/STUDY/LAB/pool.json"))
    parser.add_argument("--write-only", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9]+", args.account):
        raise ValueError("invalid account")
    slug = f"dual-ar-v2-{args.codec}-pilot"
    target = REPO / "ops/_push" / args.account / slug
    target.mkdir(parents=True, exist_ok=True)
    payload = notebook(render_cell(args.commit, args.codec), args.codec)
    payload["cells"][0]["id"] = f"dual-ar-v2-{args.codec}"
    (target / "notebook.ipynb").write_text(json.dumps(payload), encoding="utf-8")
    (target / "kernel-metadata.json").write_text(json.dumps(metadata(args.account, slug)), encoding="utf-8")
    handle = f"{args.account}/{slug}"
    print(f"[generated] {handle} commit={args.commit}", flush=True)
    if args.write_only:
        return
    env = account_environment(args.pool, args.account)
    env["PYTHONIOENCODING"] = "utf-8"
    require_inactive(handle, env)
    response = subprocess.run(kaggle_command() + ["kernels", "push", "-p", str(target)],
                              capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    output = (response.stdout + response.stderr).strip()
    print(output, flush=True)
    if response.returncode or "successfully pushed" not in output.lower():
        raise SystemExit(response.returncode or 1)


if __name__ == "__main__":
    main()
