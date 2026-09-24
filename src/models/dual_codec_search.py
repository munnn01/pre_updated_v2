"""Dual-analyzer selectors. Inference reads an explicit label-free allowlist."""
from __future__ import annotations

import math
import numpy as np

from src.models.codec_search import CANDIDATES, choose_action

MODELS = ("r2plus1d_18", "r3d_18")
SIGNALS = (
    "kl_source", "feature_distance", "source_confidence", "source_margin",
    "source_top1_agrees", "confidence", "margin", "entropy",
    "source_top1_prob", "kl_anchor", "anchor_top1_agrees",
)


def observations(candidates):
    """Drop ALL outcome/label fields before any policy sees candidates."""
    if not candidates or candidates[0]["name"] != "identity128":
        raise ValueError("identity128 must be first")
    result = []
    for row in candidates:
        if row["name"] not in CANDIDATES or not math.isfinite(row["bpp"]) or row["bpp"] <= 0:
            raise ValueError("invalid candidate/rate")
        signals = {model: {key: row["signals"][model][key] for key in SIGNALS} for model in MODELS}
        if not all(math.isfinite(float(v)) for s in signals.values() for v in s.values()):
            raise ValueError("non-finite analyzer signal")
        result.append({"name": row["name"], "bpp": row["bpp"], "signals": signals})
    return result


def v1_choice(obs):
    return choose_action([{**r["signals"][MODELS[0]], "name": r["name"], "bpp": r["bpp"]}
                          for r in obs], .1, .05)


def passes(candidate, anchor, model, limits):
    c, a = candidate["signals"][model], anchor["signals"][model]
    return (c["kl_source"] <= a["kl_source"] + limits["kl"]
            and c["feature_distance"] <= a["feature_distance"] + limits["feature"]
            and (c["source_confidence"] < limits["confidence"] or c["source_top1_agrees"]))


def risk_features(obs, i, qp):
    """No target class, target probability, correctness or video ID is exposed."""
    candidate, anchor = obs[i], obs[0]
    values = [qp / 50., candidate["bpp"] / anchor["bpp"]]
    values.extend(float(candidate["name"] == name) for name in CANDIDATES[1:])
    for model in MODELS:
        c, a = candidate["signals"][model], anchor["signals"][model]
        values.extend(float(c[k]) for k in SIGNALS)
        values.extend(float(a[k]) for k in ("confidence", "margin", "entropy", "source_top1_agrees"))
        values.extend((c["kl_source"] - a["kl_source"], c["feature_distance"] - a["feature_distance"]))
    return np.asarray(values, dtype=np.float64)


def risk_scores(obs, qp, state):
    if state is None or state.get("schema") != 1:
        raise ValueError("learned gate requires fitted risk state")
    x = np.stack([risk_features(obs, i, qp) for i in range(len(obs))])
    scores = []
    for model in MODELS:
        s = state["models"][model]
        if "constant" in s:
            scores.append(np.full(len(obs), s["constant"]))
        else:
            z = ((x - np.asarray(s["mean"])) / np.asarray(s["scale"])) @ np.asarray(s["coef"]) + s["intercept"]
            scores.append(1 / (1 + np.exp(-np.clip(z, -60, 60))))
    result = np.stack(scores, axis=1)
    result[0] = 0  # identity cannot break itself
    return result


def select_observations(obs, qp, policy, risks=None):
    """One selected stream serves BOTH analyzers, independently of test labels."""
    mode = policy["mode"]
    if mode == "identity":
        return 0
    if mode == "v1":
        return v1_choice(obs)
    if mode not in ("A", "B", "C"):
        raise ValueError(f"unknown mode: {mode}")
    if mode == "A":
        i = v1_choice(obs)
        return i if passes(obs[i], obs[0], MODELS[1], policy["cross"]) else 0
    feasible = [0]
    for i, row in enumerate(obs[1:], 1):
        if row["bpp"] >= obs[0]["bpp"]:
            continue
        if not passes(row, obs[0], MODELS[0], policy["primary"]):
            continue
        if mode == "B" and not passes(row, obs[0], MODELS[1], policy["cross"]):
            continue
        if mode == "C":
            if risks is None:
                raise ValueError("risk scores required")
            threshold = policy["risk_low_qp"] if qp <= 35 else policy["risk_high_qp"]
            if not np.all(np.isfinite(risks[i])) or np.any(risks[i] > threshold):
                continue
        feasible.append(i)
    return min(feasible, key=lambda i: obs[i]["bpp"])


def select(candidates, qp, policy, state=None):
    obs = observations(candidates)
    risks = risk_scores(obs, qp, state) if policy["mode"] == "C" else None
    return select_observations(obs, qp, policy, risks)


def policy_grid():
    primary = {"kl": .1, "feature": .05, "confidence": .6}
    grid = []
    for kl, feature in ((0., 0.), (.02, .005), (.05, .02), (.1, .05)):
        for confidence in (.6, .8, 1.01):  # 1.01 explicitly disables source argmax guard
            cross = {"kl": kl, "feature": feature, "confidence": confidence}
            grid.append({"mode": "A", "cross": cross})
            grid.append({"mode": "B", "primary": primary, "cross": cross})
    for low in (.02, .05, .10, .20):
        for high in (.02, .05, .10, .20):
            grid.append({"mode": "C", "primary": primary,
                         "risk_low_qp": low, "risk_high_qp": high})
    return grid
