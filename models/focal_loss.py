"""
focal_loss.py — Focal Loss cho xử lý mất cân bằng nhãn cực độ

Driver genes (~5-10%) vs Passenger genes (~90-95%) → Mất cân bằng nghiêm trọng.
Focal Loss giảm ảnh hưởng của các mẫu dễ phân loại, tập trung vào hard examples.

Công thức (Lin et al. 2017):
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    alpha: Cân bằng trọng số lớp (alpha cho lớp positive, 1-alpha cho negative)
    gamma: Focusing parameter (gamma=0 → Cross Entropy, gamma=2 → mặc định)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Binary Focal Loss cho bài toán phân loại Driver/Passenger gene.

    Args:
        alpha:  Trọng số lớp positive (driver gene). Mặc định 0.75.
                (Cao vì driver gene là lớp minority)
        gamma:  Focusing exponent. Mặc định 2.0.
        reduction: "mean", "sum", hoặc "none"
    """

    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs:  Logits từ mô hình, shape (N,) hoặc (N, 1)
            targets: Nhãn nhị phân 0/1, shape (N,)
        Returns:
            Focal loss scalar (nếu reduction="mean")
        """
        inputs = inputs.squeeze(-1).float()
        targets = targets.float()

        # Binary Cross Entropy (không reduce)
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none"
        )

        # Tính p_t: xác suất predicted cho nhãn đúng
        probs = torch.sigmoid(inputs)
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Tính alpha_t: trọng số lớp
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Focal Loss
        focal_loss = alpha_t * focal_weight * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss

    def __repr__(self) -> str:
        return (
            f"FocalLoss(alpha={self.alpha}, gamma={self.gamma}, "
            f"reduction='{self.reduction}')"
        )
