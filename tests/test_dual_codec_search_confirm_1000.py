import json
from pathlib import Path

import pytest

from ops import dual_codec_search_confirm_1000 as confirm
from ops.merge_dual_codec_search_confirm_1000 import load_shards
from ops.push_dual_codec_search_confirm_1000 import render_cell


def test_frozen_pilot_artifacts_are_exact_and_unchanged(tmp_path, monkeypatch):
    config = json.loads(confirm.CONFIG.read_text(encoding="utf-8"))
    for codec in ("h264", "h265"):
        risk, frozen, selected = confirm.load_frozen(codec, config)
        assert risk["n_fit_videos"] == 400
        assert selected == frozen["policies"]["C"]
        assert selected["mode"] == "C"
    folder = tmp_path / "h264"
    folder.mkdir()
    for name in ("risk_model.json", "frozen_policy.json"):
        (folder / name).write_bytes((confirm.ARTIFACTS / "h264" / name).read_bytes())
    monkeypatch.setattr(confirm, "ARTIFACTS", tmp_path)
    (folder / "risk_model.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        confirm.load_frozen("h264", config)


def test_notebook_is_pinned_and_no_credentials_embedded():
    cell = render_cell("a" * 40, "h265", 1)
    assert "pre_updated_v2.git" in cell and "a" * 40 in cell
    assert "--shard \"$SHARD\"" in cell
    assert "KGAT_" not in cell and "timeout " not in cell
    assert "__REF__" not in cell and "__CODEC__" not in cell and "__SHARD__" not in cell
    with pytest.raises(ValueError):
        render_cell("main", "h264", 0)
    with pytest.raises(ValueError):
        render_cell("a" * 40, "h264", 2)


def test_merger_rejects_missing_shard_before_any_metric(tmp_path):
    config = json.loads(confirm.CONFIG.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="exactly two"):
        load_shards([Path(tmp_path)], "h264", config)
