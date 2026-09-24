"""Small deterministic checks for the post-hoc paper-validation reporter."""

import numpy as np
import pytest

from ops.paper_validation import fixed_arrays, markdown
from src.models.codec_search import CANDIDATES
from src.models.dual_codec_search import MODELS


def test_fixed_arrays_uses_same_candidate_at_every_qp():
    def measurement(qp):
        return {"qp": qp, "candidates": [
            {"name": name, "bpp": (i + 1) / 10, "correct": i % 2 == 0,
             "cross_correct": i % 2 == 1}
            for i, name in enumerate(CANDIDATES)
        ]}

    rows = [{"measurements": [measurement(30), measurement(35)]} for _ in range(3)]
    values, choices = fixed_arrays(rows, "area96")
    assert values.shape == (3, 2, 3)
    assert np.all(values[:, :, 0] == pytest.approx(.3))
    assert np.all(values[:, :, 1] == 1.)
    assert np.all(values[:, :, 2] == 0.)
    assert choices == {"area96": 6}


def test_fixed_arrays_rejects_invalid_or_reordered_candidate():
    with pytest.raises(ValueError, match="unknown fixed"):
        fixed_arrays([], "not_a_candidate")
    rows = [{"measurements": [{"candidates": [
        {"name": "wrong", "bpp": .1, "correct": True, "cross_correct": True}
    ]}]}]
    with pytest.raises(ValueError, match="candidate order"):
        fixed_arrays(rows, "identity128")


def test_markdown_reports_both_models_and_exploratory_warning():
    score = {"metrics": {"bd_rate_top1_pct": -12., "bd_accuracy_top1_pp": 3.,
                         "min_same_qp_top1_gap": -.005}}
    interval = {"bd_rate_top1_pct": {"ci95": [-14., -10.]}}
    report = {"codec": "h264", "n": 1000,
              "arms": {"C": {"analyzers": {m: score for m in MODELS},
                             "bootstrap": {m: interval for m in MODELS}}},
              "C_vs_B": {"analyzers": {m: score for m in MODELS},
                         "bootstrap": {m: interval for m in MODELS}}}
    page = markdown(report)
    assert "exploratory" in page
    assert "r2plus1d_18" in page and "r3d_18" in page
    assert "-12.00" in page and "[-14.00, -10.00]" in page
