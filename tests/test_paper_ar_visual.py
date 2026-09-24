from pathlib import Path

import numpy as np
import pytest

from ops.paper_ar_visual import pick_ids, save_panel


def test_pick_ids_is_order_independent_and_preselected():
    ids = [f"class/video_{i}.mp4" for i in range(20)]
    assert pick_ids(ids, 5, "frozen") == pick_ids(list(reversed(ids)), 5, "frozen")
    assert len(pick_ids(ids, 5, "frozen")) == 5
    with pytest.raises(ValueError):
        pick_ids(ids + [ids[0]], 5, "frozen")


def test_save_panel_keeps_frames_and_writes_png(tmp_path: Path):
    src = np.zeros((16, 128, 128, 3), dtype=np.uint8)
    trial = np.zeros((16, 96, 96, 3), dtype=np.uint8)
    src[8, :, :, 0] = 255
    path = tmp_path / "figure.png"
    save_panel(src, src, trial, (4, 8, 12), path)
    assert path.is_file()
    from PIL import Image
    assert Image.open(path).size == (768, 858)
