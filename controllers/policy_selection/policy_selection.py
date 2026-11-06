#!/usr/bin/env python3
"""Webots controller entry point and offline detector trainer.

Place this folder at controllers/policy_selection and configure the TurtleBot's
controller as 'policy_selection'. Webots passes Robot.controllerArgs here.
"""
import argparse


def build_parser():
    parser = argparse.ArgumentParser(description="Policy Selection training and runtime")
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("collect", help="Capture clean independent Webots sensor sequences")
    p.add_argument("--output", required=True)
    p.add_argument("--episodes", type=int, default=80)
    p.add_argument("--steps-per-episode", type=int, default=100)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--checkpoint", default=None, help="Optional PPO to drive collection")
    p.add_argument("--policy-modality", default="image_lidar_world_info",
                   choices=("image_lidar_world_info", "image_lidar", "image", "lidar_world_info"))
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))

    p = commands.add_parser("train-detector", help="Offline training/calibration (Webots not needed)")
    p.add_argument("--modality", choices=("image", "lidar"), required=True)
    p.add_argument("--data", required=True, help="Directory containing capture manifest.json")
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--val-fraction", type=float, default=.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--max-train-frames", type=int, default=None)
    p.add_argument("--max-val-frames", type=int, default=None)

    p = commands.add_parser("train-policy", help="PPO independent observation subset training")
    p.add_argument("--modality", required=True,
                   choices=("image_lidar_world_info", "image_lidar", "image", "lidar_world_info"))
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, default=15_000_000)
    p.add_argument("--save-freq", type=int, default=2_000_000)
    p.add_argument("--n-steps", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--verbose", type=int, default=1)

    conditions = ("normal", "partial_image", "full_image", "partial_lidar", "full_lidar",
                  "partial_both", "full_both")
    p = commands.add_parser("infer-policy", help="Independent policy inference in Webots")
    p.add_argument("--modality", required=True,
                   choices=("image_lidar_world_info", "image_lidar", "image", "lidar_world_info"))
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--condition", choices=conditions, default="normal")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--deterministic", action="store_true", help="Default: stochastic PPO sampling")
    p.add_argument("--output", default=None)

    p = commands.add_parser("infer-mapek", help="Learned detectors and temporal policy selection")
    p.add_argument("--library", required=True, help="JSON with paths for all 4 PPO archives")
    p.add_argument("--image-detector", required=True)
    p.add_argument("--lidar-detector", required=True)
    p.add_argument("--condition", choices=conditions, default="normal")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--onset", type=int, default=5)
    p.add_argument("--recovery", type=int, default=10)
    p.add_argument("--dwell", type=int, default=10)
    p.add_argument("--seed", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--output", default=None)

    p = commands.add_parser("infer-detector", help="Stream learned detector logits in Webots")
    p.add_argument("--image-detector", required=True)
    p.add_argument("--lidar-detector", required=True)
    p.add_argument("--condition", choices=conditions, default="normal")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=4)
    p.add_argument("--device", default="auto")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "train-policy":
        from selection.train_policy import run
    elif args.command == "train-detector":
        from selection.train_detector import run
        if args.batch_size is None:
            args.batch_size = 4 if args.modality == "image" else 32
    elif args.command == "collect":
        from selection.dataset import collect as run
    elif args.command == "infer-policy":
        from selection.infer import run_policy as run
    elif args.command == "infer-mapek":
        from selection.infer import run_mapek as run
    elif args.command == "infer-detector":
        from selection.detect import run
    else:
        raise AssertionError(args.command)
    run(args)


if __name__ == "__main__":
    main()
