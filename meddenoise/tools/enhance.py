"""质量增强工具: CLAHE对比度增强、非锐化掩膜、伽马校正."""

import numpy as np
import cv2


def clahe_enhance(image: np.ndarray, clip_limit: float = 2.0,
                  tile_size: int = 8) -> np.ndarray:
    u8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(u8).astype(np.float64) / 255.0


def unsharp_mask(image: np.ndarray, amount: float = 0.6,
                 radius: float = 2.0) -> np.ndarray:
    blurred = cv2.GaussianBlur(image.astype(np.float32), (0, 0), radius)
    sharpened = image + amount * (image - blurred)
    return np.clip(sharpened, 0, 1)


def gamma_correct(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    return np.clip(image, 0, 1) ** gamma


ENHANCERS = {
    "clahe": clahe_enhance,
    "unsharp": unsharp_mask,
    "gamma": gamma_correct,
}


def enhance(image: np.ndarray, method: str, **kwargs) -> np.ndarray:
    if method not in ENHANCERS:
        raise ValueError(f"未知增强方法 '{method}', 可选: {list(ENHANCERS)}")
    return ENHANCERS[method](image, **kwargs)
