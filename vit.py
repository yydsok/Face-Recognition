import torch
import torch.nn as nn
from torch import Tensor
from typing import Any
from torchvision.models import vit_b_16

class ViT(nn.Module):
    def __init__(self, num_classes: int = 1000, image_size: int = 224, patch_size: int = 16, dim: int = 768, depth: int = 12, heads: int = 12, mlp_dim: int = 3072, dropout: float = 0.1, emb_dropout: float = 0.1):
        super().__init__()
        assert image_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
            nn.Unflatten(2, (num_patches, dim))
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = nn.Sequential(*[
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout) for _ in range(depth)
        ])
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape
        x = x + self.pos_embedding[:, 1:(n + 1)]
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.dropout(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)
        x = x.transpose(0, 1)
        x = self.mlp_head(x[:, 0])
        return x






# Load a pre-trained ViT model
#model = vit_b_16(weights='IMAGENET1K_V1')
