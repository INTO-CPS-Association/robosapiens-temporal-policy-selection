import numpy as np
from selection.mapek import TemporalSelector, MAPEKController
from selection.modalities import observation_for


def test_healthy_preferred_and_safety_fallback():
    s = TemporalSelector(onset=2, recovery=3, dwell=10)
    assert s.update(False, False)["active_policy"] == "image_lidar_world_info"
    assert s.update(True, True)["active_policy"] == "image_lidar_world_info"
    decision = s.update(True, True)
    assert decision["active_policy"] == "safe"
    assert decision["health"] == {"image": False, "lidar": False, "world_info": False}


def test_dwell_never_retains_inadmissible_policy():
    s = TemporalSelector(onset=2, recovery=2, dwell=100)
    s.update(False, False)
    s.update(True, False)
    assert s.update(True, False)["active_policy"] == "lidar_world_info"
    s.update(False, False)
    d = s.update(False, False)
    assert d["requested_policy"] == "image_lidar_world_info"
    assert d["active_policy"] == "lidar_world_info"  # switchback is dwell-constrained
    for _ in range(99):
        d = s.update(False, False)
    assert d["active_policy"] == "image_lidar_world_info"


def test_lidar_world_dependency():
    s = TemporalSelector(onset=1, recovery=1, dwell=0)
    d = s.update(False, True)
    assert d["active_policy"] == "image"
    assert "world_info" not in [k for k, v in d["health"].items() if v]


def test_image_only_unwrapped():
    sample = dict(image=np.zeros((270,480,3), dtype=np.uint8), lidar=np.zeros((8,1200)),
                  world_info=np.zeros(3))
    assert isinstance(observation_for("image", sample), np.ndarray)
    assert set(observation_for("lidar_world_info", sample)) == {"lidar", "world_info"}


class _Monitor:
    def analyze(self, observation):
        return {"image_raw": True, "lidar_raw": False}


class _Policy:
    def predict(self, observation, deterministic=False):
        return np.array(2), None


def test_controller_routes_observations():
    from selection.modalities import POLICY_INPUTS
    policies = {k: _Policy() for k in POLICY_INPUTS}
    c = MAPEKController(_Monitor(), policies, TemporalSelector(onset=1, recovery=1))
    sample = dict(image=np.zeros((270,480,3), dtype=np.uint8), lidar=np.zeros((8,1200)),
                  world_info=np.zeros(3))
    action, info = c.act(sample)
    assert action == 2
    assert info["active_policy"] == "lidar_world_info"
