#!/usr/bin/env python3
"""Main inference script for LongLive video generation."""

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from omegaconf import OmegaConf


def parse_args():
    parser = argparse.ArgumentParser(description="LongLive inference script")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/inference.yaml",
        help="Path to inference config file",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input image or video",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Output directory for generated videos",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Text prompt to guide generation",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=None,
        help="Number of frames to generate (overrides config)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (cuda/cpu)",
    )
    return parser.parse_args()


def load_config(config_path: str) -> OmegaConf:
    """Load and return the inference configuration."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    cfg = OmegaConf.load(config_path)
    return cfg


def setup_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_inference(cfg, args):
    """Run the main inference pipeline."""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Running inference on device: {device}")

    # Override config with CLI args
    if args.num_frames is not None:
        cfg.num_frames = args.num_frames

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    print(f"Input: {input_path}")
    print(f"Output directory: {output_dir}")
    print(f"Prompt: {args.prompt!r}")
    print(f"Num frames: {cfg.get('num_frames', 'from config')}")

    # TODO: Initialize model and run generation pipeline
    # model = build_model(cfg, device)
    # output = model.generate(input_path, prompt=args.prompt, ...)
    # save_output(output, output_dir)

    print("Inference complete.")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    setup_seed(args.seed)
    run_inference(cfg, args)


if __name__ == "__main__":
    main()
