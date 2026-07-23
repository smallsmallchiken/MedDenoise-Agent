"""图像感知分析工具: 噪声估计、结构分析、模态识别."""

import numpy as np
from scipy import ndimage
from skimage.restoration import estimate_sigma
from skimage.feature import structure_tensor, structure_tensor_eigenvalues
from skimage.filters import sobel


def estimate_noise_sigma(image: np.ndarray) -> float:
    """基于小波系数中值绝对偏差(MAD)估计高斯噪声标准差(0-255尺度)."""
    sigma = estimate_sigma(image, channel_axis=None, average_sigmas=True)
    return float(sigma * 255.0)


def structure_analysis(image: np.ndarray) -> dict:
    """结构张量分析: 计算边缘密度、纹理复杂度与各向异性(相干性)."""
    Axx, Axy, Ayy = structure_tensor(image, sigma=1.5)
    l1, l2 = structure_tensor_eigenvalues([Axx, Axy, Ayy])
    coherence = np.where(l1 + l2 > 1e-12, ((l1 - l2) / (l1 + l2 + 1e-12)) ** 2, 0.0)
    edges = sobel(image)
    edge_density = float(np.mean(edges > 0.1))
    local_var = ndimage.generic_filter(image, np.var, size=7)
    return {
        "edge_density": edge_density,
        "mean_coherence": float(np.mean(coherence)),
        "texture_complexity": float(np.mean(local_var)),
        "dynamic_range": float(image.max() - image.min()),
    }


def guess_modality(image: np.ndarray, filename: str = "") -> str:
    """基于文件名与灰度分布的启发式模态识别 (CT/MRI/X-Ray/US)."""
    name = filename.lower()
    for key, modality in [("ct", "CT"), ("mri", "MRI"), ("mr", "MRI"),
                          ("xray", "X-Ray"), ("x-ray", "X-Ray"),
                          ("us", "Ultrasound"), ("ultra", "Ultrasound")]:
        if key in name:
            return modality
    hist, _ = np.histogram(image, bins=64, range=(0, 1))
    hist = hist / hist.sum()
    background_ratio = float(hist[:4].sum())
    if background_ratio > 0.45:
        return "CT"
    if float(np.std(image)) > 0.28:
        return "X-Ray"
    return "MRI"


def analyze_image(image: np.ndarray, filename: str = "") -> dict:
    """综合感知分析: 供感知Agent调用的一站式工具."""
    sigma = estimate_noise_sigma(image)
    structure = structure_analysis(image)
    modality = guess_modality(image, filename)
    if sigma < 5:
        noise_level = "低"
    elif sigma < 20:
        noise_level = "中等"
    else:
        noise_level = "高"

    edge_density = structure["edge_density"]
    texture_complexity = structure["texture_complexity"]
    dynamic_range = structure["dynamic_range"]
    mean_coherence = structure["mean_coherence"]

    if noise_level == "高" or sigma >= 25:
        recommendation = "建议采用强滤波或 Agent 调参，先充分抑制噪声再保护细节。"
    elif edge_density > 0.15 and texture_complexity > 0.02:
        recommendation = "边缘与纹理丰富，建议提高纹理增强权重，避免过度平滑。"
    elif edge_density < 0.08:
        recommendation = "图像整体较平滑，可用标准参数或略强的基础滤波。"
    else:
        recommendation = "建议直接使用 VT-BM3D 标准流程，必要时由 Agent 微调。"

    summary = (
        f"该图像被识别为 {modality} 影像，噪声水平为 {noise_level}（估计 σ={sigma:.2f}）。"
        f"边缘密度 {edge_density:.3f}，纹理复杂度 {texture_complexity:.4f}，方向一致性 {mean_coherence:.3f}，"
        f"动态范围 {dynamic_range:.3f}。{recommendation}"
    )

    return {
        "estimated_sigma": round(sigma, 2),
        "noise_level": noise_level,
        "modality": modality,
        "summary": summary,
        "recommendation": recommendation,
        **{k: round(v, 4) for k, v in structure.items()},
    }
