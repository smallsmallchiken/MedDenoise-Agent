"""质量评价工具: 全参考(PSNR/SSIM/RMSE)与无参考指标."""

import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr(reference: np.ndarray, target: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(reference, target, data_range=1.0))


def ssim(reference: np.ndarray, target: np.ndarray) -> float:
    return float(structural_similarity(reference, target, data_range=1.0))


def rmse(reference: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((reference - target) ** 2)))


def full_reference_metrics(reference: np.ndarray, target: np.ndarray) -> dict:
    return {
        "psnr": round(psnr(reference, target), 2),
        "ssim": round(ssim(reference, target), 4),
        "rmse": round(rmse(reference, target), 4),
    }


def no_reference_metrics(image: np.ndarray) -> dict:
    """无参考指标: 拉普拉斯清晰度、梯度能量、估计噪声、熵."""
    u8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    lap_var = float(cv2.Laplacian(u8, cv2.CV_64F).var())
    gx = cv2.Sobel(u8, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(u8, cv2.CV_64F, 0, 1)
    grad_energy = float(np.mean(gx ** 2 + gy ** 2))
    hist, _ = np.histogram(u8, bins=256, range=(0, 255))
    p = hist / hist.sum()
    entropy = float(-np.sum(p[p > 0] * np.log2(p[p > 0])))
    return {
        "laplacian_sharpness": round(lap_var, 2),
        "gradient_energy": round(grad_energy, 2),
        "entropy": round(entropy, 4),
    }
