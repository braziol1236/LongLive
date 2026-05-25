"""Loss functions for LongLive video generation model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ReconstructionLoss(nn.Module):
    """Pixel-level reconstruction loss with optional temporal weighting."""

    def __init__(self, loss_type: str = "l2", temporal_weight_decay: float = 1.0):
        """
        Args:
            loss_type: One of 'l1', 'l2', or 'huber'.
            temporal_weight_decay: Decay factor applied to later frames.
                                   1.0 means uniform weighting.
        """
        super().__init__()
        assert loss_type in ("l1", "l2", "huber"), f"Unknown loss type: {loss_type}"
        self.loss_type = loss_type
        self.temporal_weight_decay = temporal_weight_decay

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pred:   (B, T, C, H, W) predicted frames.
            target: (B, T, C, H, W) ground-truth frames.
            mask:   Optional (B, T, 1, H, W) binary mask (1 = valid region).

        Returns:
            Scalar loss tensor.
        """
        B, T, C, H, W = pred.shape

        if self.loss_type == "l1":
            loss = F.l1_loss(pred, target, reduction="none")
        elif self.loss_type == "l2":
            loss = F.mse_loss(pred, target, reduction="none")
        else:  # huber
            loss = F.huber_loss(pred, target, reduction="none")

        # Apply temporal decay weights so earlier frames matter more
        if self.temporal_weight_decay < 1.0:
            weights = torch.tensor(
                [self.temporal_weight_decay ** t for t in range(T)],
                device=pred.device,
                dtype=pred.dtype,
            )  # (T,)
            weights = weights.view(1, T, 1, 1, 1)
            loss = loss * weights

        if mask is not None:
            loss = loss * mask
            return loss.sum() / (mask.sum() * C + 1e-8)

        return loss.mean()


class PerceptualLoss(nn.Module):
    """Simple VGG-based perceptual loss operating on individual frames."""

    def __init__(self, feature_layers: tuple = (3, 8, 15)):
        super().__init__()
        import torchvision.models as tvm

        vgg = tvm.vgg16(weights=tvm.VGG16_Weights.DEFAULT)
        self.blocks = nn.ModuleList()
        prev = 0
        for idx in feature_layers:
            self.blocks.append(nn.Sequential(*list(vgg.features[prev:idx + 1])))
            prev = idx + 1

        # Freeze VGG weights
        for p in self.parameters():
            p.requires_grad_(False)

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize from [-1, 1] to ImageNet stats."""
        x = (x + 1.0) / 2.0  # [-1,1] -> [0,1]
        return (x - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   (B, T, C, H, W) or (B, C, H, W).
            target: same shape as pred.

        Returns:
            Scalar perceptual loss.
        """
        if pred.dim() == 5:
            B, T, C, H, W = pred.shape
            pred = pred.view(B * T, C, H, W)
            target = target.view(B * T, C, H, W)

        pred = self._normalize(pred)
        target = self._normalize(target)

        loss = torch.tensor(0.0, device=pred.device)
        for block in self.blocks:
            pred = block(pred)
            with torch.no_grad():
                target = block(target)
            loss = loss + F.l1_loss(pred, target)

        return loss


class CombinedLoss(nn.Module):
    """Weighted combination of reconstruction and perceptual losses."""

    def __init__(
        self,
        recon_weight: float = 1.0,
        perceptual_weight: float = 0.1,
        loss_type: str = "l2",
        temporal_weight_decay: float = 1.0,
        use_perceptual: bool = True,
    ):
        super().__init__()
        self.recon_weight = recon_weight
        self.perceptual_weight = perceptual_weight
        self.use_perceptual = use_perceptual

        self.recon_loss = ReconstructionLoss(loss_type, temporal_weight_decay)
        if use_perceptual:
            self.perceptual_loss = PerceptualLoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Returns:
            dict with keys 'total', 'recon', and optionally 'perceptual'.
        """
        recon = self.recon_loss(pred, target, mask)
        out = {"recon": recon, "total": self.recon_weight * recon}

        if self.use_perceptual:
            perc = self.perceptual_loss(pred, target)
            out["perceptual"] = perc
            out["total"] = out["total"] + self.perceptual_weight * perc

        return out
