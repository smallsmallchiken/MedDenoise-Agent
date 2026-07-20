"""去噪算法工具库: 经典滤波、NLM、小波、BM3D 及结构感知SA-BM3D."""

import numpy as np
import cv2
from skimage.restoration import denoise_nl_means, denoise_wavelet, estimate_sigma
from skimage.filters import sobel

try:
    import bm3d as _bm3d

    HAS_BM3D = True
except ImportError:  # pragma: no cover
    HAS_BM3D = False


def _to_uint8(image: np.ndarray) -> np.ndarray:
    return (np.clip(image, 0, 1) * 255).astype(np.uint8)


def _to_float(image: np.ndarray) -> np.ndarray:
    return image.astype(np.float64) / 255.0


def gaussian_denoise(image: np.ndarray, sigma: float = 15.0) -> np.ndarray:
    ksize = max(3, int(2 * round(sigma / 10) + 1))
    return _to_float(cv2.GaussianBlur(_to_uint8(image), (ksize, ksize), sigma / 30))


def median_denoise(image: np.ndarray, sigma: float = 15.0) -> np.ndarray:
    ksize = 3 if sigma < 35 else 5
    return _to_float(cv2.medianBlur(_to_uint8(image), ksize))


def nlm_denoise(image: np.ndarray, sigma: float = 15.0) -> np.ndarray:
    s = sigma / 255.0
    return denoise_nl_means(image, h=1.15 * s, sigma=s, fast_mode=True,
                            patch_size=7, patch_distance=11)


def wavelet_denoise(image: np.ndarray, sigma: float = 15.0) -> np.ndarray:
    return denoise_wavelet(image, sigma=sigma / 255.0, mode="soft",
                           wavelet="db4", rescale_sigma=True)


def bm3d_denoise(image: np.ndarray, sigma: float = 15.0) -> np.ndarray:
    if not HAS_BM3D:
        return nlm_denoise(image, sigma)
    return np.clip(_bm3d.bm3d(image, sigma_psd=sigma / 255.0,
                              stage_arg=_bm3d.BM3DStages.ALL_STAGES), 0, 1)


def sa_bm3d_denoise(image: np.ndarray, sigma: float = 15.0,
                    alpha: float = 0.4, beta: float = 0.6,
                    texture_boost: float = 1.4) -> np.ndarray:
    """结构感知SA-BM3D (借鉴VT-BM3D思想的Python实现).

    利用局部方差与结构张量相干性构建结构显著性图 S,
    对平坦区采用较强去噪(高sigma)、对边缘/纹理区采用较弱去噪(低sigma),
    再按 S 逐像素融合两次BM3D结果, 在提升PSNR的同时保留结构细节.
    """
    from skimage.feature import structure_tensor, structure_tensor_eigenvalues
    from scipy import ndimage

    local_var = ndimage.uniform_filter(image ** 2, 7) - ndimage.uniform_filter(image, 7) ** 2
    var_map = local_var / (local_var.max() + 1e-12)

    Axx, Axy, Ayy = structure_tensor(image, sigma=1.5)
    l1, l2 = structure_tensor_eigenvalues([Axx, Axy, Ayy])
    coherence = np.where(l1 + l2 > 1e-12, ((l1 - l2) / (l1 + l2 + 1e-12)) ** 2, 0.0)

    saliency = alpha * var_map + beta * coherence
    saliency = ndimage.gaussian_filter(saliency, 2.0)
    saliency = saliency / (saliency.max() + 1e-12)

    strong = bm3d_denoise(image, sigma * 1.15)
    weak = bm3d_denoise(image, sigma / texture_boost)
    fused = (1.0 - saliency) * strong + saliency * weak
    return np.clip(fused, 0, 1)


DENOISERS = {
    "gaussian": gaussian_denoise,
    "median": median_denoise,
    "nlm": nlm_denoise,
    "wavelet": wavelet_denoise,
    "bm3d": bm3d_denoise,
    "sa-bm3d": sa_bm3d_denoise,
}


def denoise(image: np.ndarray, method: str, sigma: float = 15.0) -> np.ndarray:
    if method not in DENOISERS:
        raise ValueError(f"未知去噪方法 '{method}', 可选: {list(DENOISERS)}")
    return DENOISERS[method](image, sigma)
