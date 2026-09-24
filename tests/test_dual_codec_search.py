import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from ops.dual_codec_search import (QPS, bootstrap, calibrate, digest, fit_risk, metrics,
                                   prepare, selected_arrays, source_id, split_plan, validate_row)
from ops.push_dual_codec_search import render_cell, require_inactive
from src.models.codec_search import CANDIDATES
from src.models.dual_codec_search import (MODELS, SIGNALS, observations, policy_grid,
                                         risk_scores, select)


def candidates():
    result = []
    for i, name in enumerate(CANDIDATES):
        signal = dict(zip(SIGNALS, (.05, .05, .9, .5, True, .8, .4, .5, .8, .01, True)))
        result.append({"name": name, "bpp": 1 - .1 * i,
                       "signals": {model: dict(signal) for model in MODELS},
                       "correct": True, "cross_correct": True})
    return result


def test_label_free_all_arms_and_risk_allowlist():
    rows = candidates()
    state = {"schema": 1, "models": {m: {"constant": .01} for m in MODELS}}
    altered = copy.deepcopy(rows)
    for row in altered:
        row["correct"] = False
        row["cross_correct"] = False
        row["target_prob"] = float("nan")
        row["label"] = 123
    assert observations(rows) == observations(altered)
    for policy in [{"mode": "v1"}, *policy_grid()]:
        assert select(rows, 30, policy, state) == select(altered, 30, policy, state)


def test_a_falls_back_b_chooses_alternative():
    rows = candidates()
    rows[-1]["signals"][MODELS[1]]["source_top1_agrees"] = False
    cross = {"kl": .1, "feature": .05, "confidence": .6}
    assert select(rows, 35, {"mode": "v1"}) == 5
    assert select(rows, 35, {"mode": "A", "cross": cross}) == 0
    assert select(rows, 35, {"mode": "B", "primary": cross, "cross": cross}) == 4


def test_qp_conditioning_and_both_risk_constraints():
    rows = candidates()
    policy = {"mode": "C", "primary": {"kl": .1, "feature": .05, "confidence": .6},
              "risk_low_qp": .02, "risk_high_qp": .1}
    state = {"schema": 1, "models": {MODELS[0]: {"constant": .01}, MODELS[1]: {"constant": .05}}}
    assert select(rows, 30, policy, state) == 0
    assert select(rows, 40, policy, state) == 5
    with pytest.raises(ValueError):
        select(rows, 40, policy)


def test_invalid_signals_and_anchor_rejected():
    rows = candidates()
    rows[1]["signals"][MODELS[1]]["kl_source"] = float("nan")
    with pytest.raises(ValueError):
        observations(rows)
    with pytest.raises(ValueError):
        observations(candidates()[1:])


def test_disjoint_source_plan_balanced_and_repeatable():
    def record(i):
        return {"path": f"/dataset/class{i % 4}/clip{i}.mp4", "label": i % 4}
    index = {"train": [record(i) for i in range(120)],
             "val": [record(i) for i in range(120, 160)],
             "test": [record(i) for i in range(160, 180)]}
    # Also a duplicate of a validation source in TRAIN, which must be excluded.
    index["train"].append(index["val"][0])
    plan = split_plan(index, 40, 40, 20)
    assert plan == split_plan(index, 40, 40, 20)
    assert not set(plan["fit"]) & set(plan["calibration"])
    assert 120 not in plan["fit"] + plan["calibration"]
    with pytest.raises(ValueError):
        split_plan(index, 100, 100, 20)
    assert source_id({"path": "/class/abcdefghijk_000010_000020.mp4"}) == "abcdefghijk"
    assert source_id({"path": "/other/abcdefghijk_000100_000110.mp4"}) == "abcdefghijk"


def fake_rows(n=40):
    rows = []
    for i in range(n):
        measurements = []
        for j, qp in enumerate(QPS):
            cs = candidates()
            for k, c in enumerate(cs):
                c["bpp"] *= 1 - j * .15
                c["correct"] = i % 10 < 9 - j
                c["cross_correct"] = i % 10 < 8 - j
                if k and i % 7 == 0:
                    c["correct"] = c["cross_correct"] = False
                    for signal in c["signals"].values():
                        signal["margin"] = .01
            measurements.append({"qp": qp, "candidates": cs})
        rows.append({"sequence_id": f"clip{i}", "measurements": measurements})
    return rows


def test_fit_export_reload_calibrate_evaluate_and_bootstrap():
    rows = fake_rows()
    state = fit_risk(rows[:20])
    reloaded = json.loads(json.dumps(state, allow_nan=False))
    obs = observations(rows[0]["measurements"][0]["candidates"])
    assert np.allclose(risk_scores(obs, 30, state), risk_scores(obs, 30, reloaded))
    prepared = prepare(rows[20:], reloaded)
    calibration = calibrate(prepared)
    assert len(calibration["grid"]) == 40
    assert set(calibration["policies"]) == {"A", "B", "C"}
    base, _ = selected_arrays(prepared, {"mode": "identity"})
    trial, _ = selected_arrays(prepared, calibration["selected_policy"])
    scores = metrics(base, trial)
    assert set(scores) == set(MODELS)
    intervals = bootstrap(base, trial, 5)
    assert intervals[MODELS[0]]["bd_accuracy_top1_pp"]["requested_draws"] == 5
    for model in MODELS:
        metric = scores[model]["metrics"]
        assert metric["bd_accuracy_top1_pp"] == pytest.approx(100 * metric["bd_accuracy_top1"])


def test_single_class_risk_training_fails_conservatively():
    rows = fake_rows(2)
    for row in rows:
        for measurement in row["measurements"]:
            for c in measurement["candidates"]:
                c["correct"] = c["cross_correct"] = True
    state = fit_risk(rows)
    assert all(s["constant"] == 1 for s in state["models"].values())


def test_cache_manifest_and_candidate_completeness():
    row = {**fake_rows(1)[0], "schema": 3, "codec": "h264", "cache_key": "abc"}
    validate_row(row, "clip0", "h264", "abc")
    with pytest.raises(ValueError):
        validate_row(row, "clip0", "h264", "wrong")
    row["measurements"][0]["candidates"].pop()
    with pytest.raises(ValueError):
        validate_row(row, "clip0", "h264", "abc")


def test_notebook_pins_v2_repo_and_has_no_token_or_timeout():
    cell = render_cell("a" * 40, "h264")
    assert "pre_updated_v2.git" in cell and "a" * 40 in cell
    assert "__REF__" not in cell and "__CODEC__" not in cell
    assert "KGAT_" not in cell and "timeout " not in cell
    assert 'mkdir -p "$ROOT_OUT"' in cell
    with pytest.raises(ValueError):
        render_cell("main", "h264")


@pytest.mark.parametrize("message,code,allowed", [
    ("not found (404)", 1, True), ("complete", 0, True),
    ("RUNNING", 0, False), ("queued", 0, False),
    ("401 unauthorized", 1, False), ("mysterious response", 0, False),
])
def test_duplicate_push_guard(monkeypatch, message, code, allowed):
    monkeypatch.setattr("ops.push_dual_codec_search.subprocess.run",
                        lambda *a, **kw: SimpleNamespace(stdout=message, stderr="", returncode=code))
    if allowed:
        require_inactive("account/pilot", {})
    else:
        with pytest.raises(RuntimeError):
            require_inactive("account/pilot", {})


@pytest.mark.parametrize("listing,allowed", [
    ("ref,title\naccount/older,older\n", True),
    ("ref,title\naccount/pilot,pilot\n", False),
    ("ref,title\nanother/older,older\n", False),
    ("ref,title\n", False),
])
def test_permission_denied_only_allows_proven_absent_owned_slug(monkeypatch, listing, allowed):
    def run(command, **kwargs):
        if "status" in command:
            return SimpleNamespace(stdout="Permission 'kernels.get' was denied", stderr="", returncode=1)
        return SimpleNamespace(stdout=listing, stderr="", returncode=0)
    monkeypatch.setattr("ops.push_dual_codec_search.subprocess.run", run)
    if allowed:
        require_inactive("account/pilot", {})
    else:
        with pytest.raises(RuntimeError):
            require_inactive("account/pilot", {})


def test_measure_collects_both_models_all_qps_candidates(monkeypatch):
    import torch
    from ops import dual_codec_search as runner
    calls = []
    def predict(model, video):
        calls.append(model)
        return torch.tensor([[1., 2.]]), torch.tensor([[1., 0.]])
    class Dataset:
        samples = [{"path": "class/clip.mp4"}]
        def __getitem__(self, i):
            return torch.zeros(3, 16, 128, 128), 1, {"sequence_id": "class/clip.mp4"}
    class Codec:
        codec = "h264"
        def _encode_decode_clip(self, candidate, qp):
            return candidate.copy(), .5
    monkeypatch.setattr(runner, "predict_and_feature", predict)
    monkeypatch.setattr("cv2.VideoCapture", lambda _: SimpleNamespace(read=lambda: (True, None), release=lambda: None))
    row = runner.measure(Dataset(), 0, dict(zip(MODELS, MODELS)), Codec())
    row["cache_key"] = "cache"
    validate_row(row, "class/clip.mp4", "h264", "cache")
    assert len(calls) == 2 + 2 * 5 * 6
    assert all(c["correct"] and c["cross_correct"]
               for m in row["measurements"] for c in m["candidates"])
