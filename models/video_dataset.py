"""Video dataset utilities for LongLive training and inference."""

import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_video


class VideoDataset(Dataset):
    """Dataset for loading video clips for long video generation training."""

    def __init__(
        self,
        data_root: str,
        num_frames: int = 16,
        frame_stride: int = 4,
        resolution: Tuple[int, int] = (256, 256),
        file_list: Optional[str] = None,
        split: str = "train",
    ):
        """
        Args:
            data_root: Root directory containing video files.
            num_frames: Number of frames to sample per clip.
            frame_stride: Stride between sampled frames.
            resolution: (height, width) to resize frames.
            file_list: Optional path to a .txt file listing video paths.
            split: One of 'train' or 'val'.
        """
        self.data_root = Path(data_root)
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.resolution = resolution
        self.split = split

        # Build file list
        if file_list is not None:
            with open(file_list, "r") as f:
                self.video_paths = [
                    self.data_root / line.strip()
                    for line in f
                    if line.strip()
                ]
        else:
            exts = {".mp4", ".avi", ".mov", ".mkv"}
            self.video_paths = [
                p for p in sorted(self.data_root.rglob("*"))
                if p.suffix.lower() in exts
            ]

        if not self.video_paths:
            raise ValueError(f"No video files found under {data_root}")

        self.transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self) -> int:
        return len(self.video_paths)

    def _load_frames(self, video_path: Path) -> Optional[torch.Tensor]:
        """Load a clip of num_frames from a video file.

        Returns:
            Tensor of shape (T, C, H, W) with values in [-1, 1], or None on failure.
        """
        try:
            # read_video returns (T, H, W, C) uint8
            frames, _, info = read_video(
                str(video_path), output_format="THWC", pts_unit="sec"
            )
        except Exception as e:
            print(f"[VideoDataset] Failed to read {video_path}: {e}")
            return None

        total_frames = frames.shape[0]
        required = self.num_frames * self.frame_stride

        if total_frames < required:
            # Repeat frames to fill the required length
            repeats = (required // total_frames) + 1
            frames = frames.repeat(repeats, 1, 1, 1)[:required]

        # Random start for training, fixed for val
        max_start = frames.shape[0] - required
        if self.split == "train":
            start = random.randint(0, max_start)
        else:
            start = max_start // 2

        indices = list(range(start, start + required, self.frame_stride))
        clip = frames[indices]  # (T, H, W, C)

        # Convert to float and rearrange to (T, C, H, W)
        clip = clip.permute(0, 3, 1, 2).float() / 255.0  # [0, 1]
        clip = torch.stack([self.transform(frame) for frame in clip])  # [-1, 1]
        return clip

    def __getitem__(self, idx: int) -> dict:
        video_path = self.video_paths[idx]
        clip = self._load_frames(video_path)

        # Fallback: try a random sample if loading fails
        attempts = 0
        while clip is None and attempts < 5:
            idx = random.randint(0, len(self) - 1)
            video_path = self.video_paths[idx]
            clip = self._load_frames(video_path)
            attempts += 1

        if clip is None:
            raise RuntimeError("Failed to load any valid video clip after 5 attempts.")

        return {
            "frames": clip,           # (T, C, H, W)
            "path": str(video_path),
        }


def build_dataloader(
    cfg: dict,
    split: str = "train",
    rank: int = 0,
    world_size: int = 1,
) -> torch.utils.data.DataLoader:
    """Construct a DataLoader from config dict."""
    dataset = VideoDataset(
        data_root=cfg["data_root"],
        num_frames=cfg.get("num_frames", 16),
        frame_stride=cfg.get("frame_stride", 4),
        resolution=tuple(cfg.get("resolution", [256, 256])),
        file_list=cfg.get("file_list", None),
        split=split,
    )

    sampler = (
        torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=(split == "train")
        )
        if world_size > 1
        else None
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.get("batch_size", 2),
        shuffle=(sampler is None and split == "train"),
        sampler=sampler,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=(split == "train"),
    )
    return loader
