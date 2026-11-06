"""Shared Webots runtime glue. Import Webots modules only when a simulation starts."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from .modalities import POLICY_INPUTS
from .config import read_json


def make_environment(modality="image_lidar_world_info", seed=1, full=False):
    from .webots_env import Env
    keys = list(POLICY_INPUTS[modality])
    if full:
        keys = ["image", "lidar", "world_info"]
    env = Env(seed=seed, action_space="discrete", observation_space=keys,
              kb_control=False, inference_mode=True if full else False)
    # The published navigation policies were trained on clean observations;
    # anomalies are controlled explicitly during the inference workflow.
    env.anomalies_enabled = False
    env.clear_experiment_anomaly()
    return env


def install_condition(env, mode: str, rng: np.random.Generator):
    """Install one stable sensor-fixed corruption for a whole Webots episode."""
    from .corruptions import image_mask, corrupt_lidar
    if mode == "normal":
        env.clear_experiment_anomaly()
        return
    image_mask_value = image_spec = lidar_spec = None
    if mode.endswith("image") or mode.endswith("both"):
        if mode.startswith("partial"):
            image_mask_value, image_spec = image_mask(270, 480, rng)
            contents = ("black", "gray", "color")
            variant = str(rng.choice(contents))
            image_spec["color"] = ([0, 0, 0] if variant == "black" else
                                   [127, 127, 127] if variant == "gray" else
                                   [int(c) for c in rng.integers(0, 256, 3)])
    if mode.endswith("lidar") or mode.endswith("both"):
        if mode.startswith("partial"):
            _, lidar_spec = corrupt_lidar(np.zeros((8, 1200), dtype=np.float32), rng)
    env.set_experiment_anomaly(mode, image_mask=image_mask_value,
                               image_spec=image_spec, lidar_spec=lidar_spec)


def model_path(config_file: str | Path, entry: str):
    base = Path(config_file).expanduser().resolve().parent
    path = Path(entry).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def load_policies(config_file: str | Path, device="auto"):
    from stable_baselines3 import PPO
    from . import policy_models  # needed for custom extractor class unpickling
    config = read_json(config_file)
    result = {}
    for name in POLICY_INPUTS:
        if name not in config:
            raise ValueError(f"Missing policy {name} in {config_file}")
        entry = config[name]
        path = model_path(config_file, entry["model_path"])
        if not path.exists() and not path.with_suffix(".zip").exists():
            raise FileNotFoundError(path)
        result[name] = PPO.load(str(path), device=device)
    return result
