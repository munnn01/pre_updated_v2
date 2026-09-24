"""CPU-only tests for paper transfer and overhead result aggregation."""

import math

from ops.codec_search_ar import QPS
from ops.paper_heldout_mc3 import curves
from ops.paper_heldout_mc3 import summarize as heldout_summary
from ops.paper_runtime import summarize as runtime_summary


def _heldout_records(n=40):
    rows = []
    for i in range(n):
        measurements = []
        for position, qp in enumerate(QPS):
            rate = .5 * (.72 ** position)
            measurements.append({"qp": qp,
                "anchor": {"bpp": rate, "correct": i < (34 - 5 * position),
                           "encode_decode_s": .1, "inference_s": .02},
                "trial": {"bpp": rate * .9, "correct": i < (35 - 5 * position),
                          "encode_decode_s": .1, "inference_s": .02}})
        rows.append({"sequence_id": f"clip-{i}", "measurements": measurements})
    return rows


def test_heldout_uses_video_as_bootstrap_unit():
    records = _heldout_records()
    anchor, trial = curves(records)
    assert all(anchor[str(qp)]["n"] == 40 for qp in QPS)
    assert trial[str(QPS[0])]["bpp"] < anchor[str(QPS[0])]["bpp"]
    report = heldout_summary(records, draws=12)
    assert report["n"] == 40
    assert math.isfinite(report["metrics"]["bd_rate_top1_pct"])
    assert report["bootstrap"]["bd_rate_top1_pct"]["requested_draws"] == 12
    assert report["timing"]["anchor_encode_decode_s"]["n"] == 40 * len(QPS)


def test_runtime_summary_reports_per_clip_overhead():
    report = runtime_summary([
        {"identity_only_s": 1., "full_selector_s": 6., "overhead_ratio": 6.},
        {"identity_only_s": 2., "full_selector_s": 10., "overhead_ratio": 5.},
    ])
    assert report["n"] == 2
    assert report["overhead_ratio"]["median"] == 5.5
    assert report["full_selector_s"]["mean"] == 8.
