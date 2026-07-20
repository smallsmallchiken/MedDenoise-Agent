"""DnCNN 深度学习去噪器 (PyTorch实现, 残差学习).

网络结构 (Zhang et al., Beyond a Gaussian Denoiser, IEEE TIP 2017):
17层全卷积网络, 首层 Conv+ReLU, 中间15层 Conv+ReLU, 末层 Conv;
网络学习预测噪声残差 v, 去噪结果为 y - v.

预训练权重来自 KAIR 模型库 (dncnn_25.pth, σ=25灰度模型),
运行 scripts/download_weights.py 自动下载至 weights/ 目录.
"""

from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False

WEIGHTS_PATH = Path(__file__).resolve().parents[2] / "weights" / "dncnn_25.pth"
_model_cache: dict = {}

if HAS_TORCH:

    class DnCNN(nn.Module):
        """17层残差去噪卷积神经网络."""

        def __init__(self, channels: int = 1, features: int = 64,
                     depth: int = 17):
            super().__init__()
            layers: list[nn.Module] = [
                nn.Conv2d(channels, features, 3, padding=1), nn.ReLU(True)]
            for _ in range(depth - 2):
                layers += [nn.Conv2d(features, features, 3, padding=1),
                           nn.ReLU(True)]
            layers.append(nn.Conv2d(features, channels, 3, padding=1))
            self.model = nn.Sequential(*layers)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return x - self.model(x)  # 残差学习: 输出 = 输入 - 预测噪声


def _load_model() -> "DnCNN | None":
    if not HAS_TORCH or not WEIGHTS_PATH.exists():
        return None
    if "model" not in _model_cache:
        model = DnCNN()
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
        model.eval()
        _model_cache["model"] = model
    return _model_cache["model"]


def is_available() -> bool:
    return HAS_TORCH and WEIGHTS_PATH.exists()


def dncnn_denoise(image: np.ndarray, sigma: float = 25.0) -> np.ndarray:
    """DnCNN去噪. 权重不可用时回退到BM3D."""
    model = _load_model()
    if model is None:
        from .denoise import bm3d_denoise

        return bm3d_denoise(image, sigma)
    with torch.no_grad():
        tensor = torch.from_numpy(image.astype(np.float32))[None, None]
        output = model(tensor)[0, 0].numpy()
    return np.clip(output.astype(np.float64), 0, 1)
