"""Train one PPO navigation policy using the original observation and action design."""
from __future__ import annotations
import random
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from .modalities import POLICY_INPUTS
from .policy_models import CustomCombinedExtractor
from .callbacks import TerminationCallback
from .runtime import make_environment
from .config import metadata, write_json, unique_directory


def linear_schedule(initial):
    return lambda remaining: remaining * initial


def run(args):
    out = unique_directory(args.output)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env = make_environment(modality=args.modality, seed=args.seed)
    wrapped = Monitor(env, filename=str(out / "monitor.csv"))
    venv = DummyVecEnv([lambda: wrapped])
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, clip_reward=10)
    policy_type = "CnnPolicy" if args.modality == "image" else "MultiInputPolicy"
    params = dict(n_steps=args.n_steps, batch_size=args.batch_size, gamma=.99,
                  gae_lambda=.95, n_epochs=10, ent_coef=.01,
                  vf_coef=.5, target_kl=.03, learning_rate=linear_schedule(args.lr),
                  policy_kwargs=dict(features_extractor_class=CustomCombinedExtractor,
                                     net_arch=dict(pi=[256, 128], vf=[256, 128]),
                                     activation_fn=torch.nn.ReLU),
                  verbose=args.verbose, seed=args.seed, device=args.device,
                  tensorboard_log=str(out / "tensorboard"))
    write_json(out / "metadata.json", metadata("ppo_policy", modality=args.modality,
                obs_keys=list(POLICY_INPUTS[args.modality]), policy_type=policy_type,
                timesteps=args.steps, seed=args.seed,
                ppo={"n_steps": args.n_steps, "batch_size": args.batch_size,
                     "gamma": .99, "gae_lambda": .95, "n_epochs": 10,
                     "ent_coef": .01, "vf_coef": .5, "target_kl": .03,
                     "lr_initial_linear": args.lr, "net_arch": {"pi": [256,128], "vf": [256,128]}},
                action_mapping={"0": "forward_0.01_m", "1": "backward_0.01_m",
                                "2": "left_1_degree", "3": "right_1_degree",
                                "4": "detect_person", "5": "no_motion"},
                vecnormalize={"norm_obs": False, "norm_reward": True}))
    try:
        model = PPO(policy_type, venv, **params)
        callbacks = [TerminationCallback()]
        if args.save_freq > 0:
            callbacks.append(CheckpointCallback(save_freq=args.save_freq,
                             save_path=str(out / "checkpoints"), name_prefix=args.modality))
        model.learn(total_timesteps=args.steps, callback=CallbackList(callbacks))
        model.save(str(out / "model.zip"))
        venv.save(str(out / "vecnormalize.pkl"))
        print(f"Saved policy and metadata to {out}")
    finally:
        venv.close()
