"""VGG-16 ImageNet features for page-classifier baselines (4096-D, penultimate FC)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models


class VGG16FeatureExtractor:
    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights = models.VGG16_Weights.IMAGENET1K_V1
        vgg = models.vgg16(weights=weights).to(self.device)
        vgg.classifier = nn.Sequential(*list(vgg.classifier.children())[:-1])
        vgg.eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self._model = vgg
        self._preprocess = weights.transforms()

    @staticmethod
    def feature_dim() -> int:
        return 4096

    def encode_paths(self, paths: list[Path], batch_size: int = 32) -> np.ndarray:
        n = len(paths)
        dim = self.feature_dim()
        X = np.empty((n, dim), dtype=np.float32)
        with torch.no_grad():
            idx = 0
            for start in range(0, n, batch_size):
                chunk = paths[start : start + batch_size]
                tensors = []
                for p in chunk:
                    with Image.open(p).convert("RGB") as im:
                        tensors.append(self._preprocess(im))
                xb = torch.stack(tensors).to(self.device)
                feat = self._model(xb).cpu().numpy().astype(np.float32)
                end = idx + feat.shape[0]
                X[idx:end] = feat
                idx = end
        return X
