"""Independent PPO and temporal MAPE-K Webots inference."""
from __future__ import annotations
import json
import numpy as np
from stable_baselines3 import PPO
from . import policy_models  # pickle compatibility
from .runtime import make_environment, install_condition, load_policies
from .modalities import observation_for
from .mapek import TemporalSelector, MAPEKController
from .monitor import LearnedMonitor
from .config import write_json, unique_directory


def _episode_log(args, rows):
    if args.output:
        dest = unique_directory(args.output)
        with (dest / "trace.jsonl").open("w", encoding="utf8") as f:
            for row in rows:
                f.write(json.dumps(row, default=str) + "\n")
        print(f"Saved runtime trace: {dest}")


def run_policy(args):
    rng = np.random.default_rng(args.seed)
    env = make_environment(args.modality, seed=args.seed)
    model = PPO.load(args.checkpoint, device=args.device)
    rows = []
    try:
        for ep in range(args.episodes):
            install_condition(env, args.condition, rng)
            obs, _ = env.reset()
            # A single-policy env already exposes only the requested subset.
            for step in range(args.max_steps):
                action, _ = model.predict(obs, deterministic=args.deterministic)
                obs, reward, terminated, truncated, info = env.step(int(action))
                if terminated or truncated:
                    rows.append({"episode": ep, "steps": step+1, "reward": float(reward),
                                 "condition": args.condition, "collided": bool(info.get("collided", False))})
                    break
        _episode_log(args, rows)
    finally:
        env.supervisor.cleanup() if hasattr(env.supervisor, "cleanup") else None


def run_mapek(args):
    rng = np.random.default_rng(args.seed)
    policies = load_policies(args.library, device=args.device)
    monitor = LearnedMonitor(args.image_detector, args.lidar_detector, device=args.device)
    selector = TemporalSelector(onset=args.onset, recovery=args.recovery, dwell=args.dwell)
    controller = MAPEKController(monitor, policies, selector, safe_action=5,
                                  deterministic=args.deterministic)
    env = make_environment(seed=args.seed, full=True)
    traces = []
    try:
        for ep in range(args.episodes):
            install_condition(env, args.condition, rng)
            obs, _ = env.reset()
            controller.reset()
            for step in range(args.max_steps):
                action, decision = controller.act(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                traces.append({"episode": ep, "step": step, "condition": args.condition,
                               "reward": float(reward), "done": bool(terminated or truncated),
                               **decision})
                if decision["switched"]:
                    print(f"[ep={ep} step={step}] {decision['previous_policy']} -> {decision['active_policy']}")
                if terminated or truncated:
                    break
        _episode_log(args, traces)
    finally:
        env.supervisor.cleanup() if hasattr(env.supervisor, "cleanup") else None
