"""The four mutually trainable observation-subset policies."""

from __future__ import annotations

from collections.abc import Mapping
import numpy as np

POLICY_INPUTS: dict[str, tuple[str, ...]] = {
    "image_lidar_world_info": ("image", "lidar", "world_info"),
    "image_lidar": ("image", "lidar"),
    "image": ("image",),
    "lidar_world_info": ("lidar", "world_info"),
}
PREFERENCE = tuple(POLICY_INPUTS)


def observation_for(policy: str, observation: Mapping[str, np.ndarray]):
    """Return raw image array for CnnPolicy, dictionaries for MultiInputPolicy."""
    required = POLICY_INPUTS[policy]
    if policy == "image":
        return np.asarray(observation["image"])
    return {key: np.asarray(observation[key]) for key in required}


def check_observation(observation: Mapping[str, np.ndarray]) -> None:
    image = np.asarray(observation["image"])
    lidar = np.asarray(observation["lidar"])
    world = np.asarray(observation["world_info"])
    if image.shape != (270, 480, 3) or image.dtype != np.uint8:
        raise ValueError(f"Expected RGB-like camera (270,480,3) uint8, got {image.shape} {image.dtype}")
    if lidar.shape != (8, 1200):
        raise ValueError(f"Expected LiDAR stack (8,1200), got {lidar.shape}")
    if world.shape != (3,):
        raise ValueError(f"Expected world info (3,), got {world.shape}")
