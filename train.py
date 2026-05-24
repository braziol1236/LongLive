#!/usr/bin/env python3
"""Training script for LongLive video generation model."""

import argparse
import logging
import os
import random

import numpy as np
import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train LongLive model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/nvfp4/train_ar_nvfp4.yaml",
        help="Path to training config file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/train",
        help="Directory to save checkpoints and logs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
        help="Local rank for distributed training",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load YAML config from the given path."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config from {config_path}")
    return config


def setup_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")


def setup_distributed():
    """Initialize distributed training if available."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        logger.info(f"Distributed training: rank={rank}, world_size={world_size}")
        return rank, world_size, local_rank
    return 0, 1, 0


def build_optimizer(model, config: dict):
    """Build optimizer from config."""
    opt_cfg = config.get("optimizer", {})
    lr = opt_cfg.get("lr", 1e-4)
    weight_decay = opt_cfg.get("weight_decay", 0.01)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    logger.info(f"Optimizer: AdamW, lr={lr}, weight_decay={weight_decay}")
    return optimizer


def save_checkpoint(model, optimizer, step: int, output_dir: str):
    """Save model and optimizer state to a checkpoint file."""
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, f"checkpoint_step{step}.pt")
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        ckpt_path,
    )
    logger.info(f"Checkpoint saved to {ckpt_path}")


def main():
    args = parse_args()
    config = load_config(args.config)
    setup_seed(args.seed)

    rank, world_size, local_rank = setup_distributed()
    is_main = rank == 0

    os.makedirs(args.output_dir, exist_ok=True)

    train_cfg = config.get("training", {})
    max_steps = train_cfg.get("max_steps", 100000)
    save_every = train_cfg.get("save_every", 1000)
    log_every = train_cfg.get("log_every", 100)

    logger.info(
        f"Training config: max_steps={max_steps}, "
        f"save_every={save_every}, log_every={log_every}"
    )
    logger.info("Model and dataloader initialization would happen here.")
    logger.info("Training loop would run here.")


if __name__ == "__main__":
    main()
