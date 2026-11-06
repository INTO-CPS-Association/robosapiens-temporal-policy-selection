import numpy as np
from selection.corruptions import image_mask, corrupt_lidar, corrupt_image, apply_condition
from selection.dataset import split_sequences
from selection.train_detector import threshold_from_validation


def test_five_rgb_shapes_and_full_failure():
    for shape in ("rectangle", "ellipse", "polygon", "blob", "strip"):
        m, spec = image_mask(270,480, np.random.default_rng(17), .2, shape)
        assert m.shape == (270,480) and m.any()
        assert spec["shape"] == shape
    src = np.full((270,480,3), 128, dtype=np.uint8)
    full, _ = corrupt_image(src, np.random.default_rng(1), full=True)
    assert (full == 0).all() and (src == 128).all()


def test_lidar_persistent_sector_and_padding():
    x = np.zeros((8,1200), dtype=np.float32)
    y, spec = corrupt_lidar(x, np.random.default_rng(5), severity=.10)
    assert y.shape == x.shape
    assert np.any(y[:,60:1140] != 0)
    assert np.array_equal(y[:, :60], y[:, 1140:])  # circular left padding
    assert np.array_equal(y[:, -60:], y[:, 60:120])
    for frame in y:
        assert np.count_nonzero(frame[60:1140]) == spec["width"]


def test_modality_specific_corruptions():
    src = {"image": np.ones((270,480,3), dtype=np.uint8)*127,
           "lidar": np.zeros((8,1200), dtype=np.float32), "world_info": np.ones(3)}
    out, _ = apply_condition(src, "partial_lidar", np.random.default_rng(0))
    assert np.array_equal(out["image"], src["image"])
    assert not np.array_equal(out["lidar"], src["lidar"])


def test_split_no_sequence_leak():
    entries = [{"sequence_id": f"s{i}", "scenario_id": f"g{i//2}",
                "file": f"{i}.npz", "frames": 3} for i in range(20)]
    train, val = split_sequences({"sequences": entries}, .2, 11)
    assert {x["scenario_id"] for x in train}.isdisjoint({x["scenario_id"] for x in val})


def test_threshold_midpoint():
    t, m = threshold_from_validation(np.array([-.9, -.4, .3, .5]), np.array([0,0,1,1]))
    assert -.4 < t < .3
    assert m["balanced_accuracy"] == 1.0
