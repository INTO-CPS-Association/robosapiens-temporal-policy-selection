import torch
from selection.detectors import PatchMILImageClassifier, PersistenceLiDARClassifier


def test_parameter_counts():
    assert sum(p.numel() for p in PatchMILImageClassifier().parameters()) == 503137
    assert sum(p.numel() for p in PersistenceLiDARClassifier().parameters()) == 591777


def test_lidar_forward_shape():
    with torch.no_grad():
        assert PersistenceLiDARClassifier()(torch.zeros(2, 8, 1200)).shape == (2,)
