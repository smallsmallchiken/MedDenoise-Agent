"""数据工具: 加载示例医学影像与噪声合成."""

import numpy as np
from pathlib import Path
from skimage import data, io, transform
from skimage.util import img_as_float

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "samples"


def load_sample_images(size: int = 256) -> dict[str, np.ndarray]:
    """加载示例影像: Shepp-Logan CT体模 + skimage内置图像."""
    images = {
        "ct_phantom": transform.resize(data.shepp_logan_phantom(), (size, size)),
        "camera": transform.resize(img_as_float(data.camera()), (size, size)),
        "cell": transform.resize(img_as_float(data.cell()), (size, size)),
    }
    if DATA_DIR.exists():
        for f in sorted(DATA_DIR.glob("*.png")):
            img = img_as_float(io.imread(f, as_gray=True))
            images[f.stem] = transform.resize(img, (size, size))
    return images


def load_image(path: str, size: int | None = None) -> np.ndarray:
    img = img_as_float(io.imread(path, as_gray=True))
    if size:
        img = transform.resize(img, (size, size))
    return np.clip(img, 0, 1)


def add_gaussian_noise(image: np.ndarray, sigma: float = 15.0,
                       seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noisy = image + rng.normal(0, sigma / 255.0, image.shape)
    return np.clip(noisy, 0, 1)


def add_rician_noise(image: np.ndarray, sigma: float = 15.0,
                     seed: int = 0) -> np.ndarray:
    """莱斯噪声: MRI幅值图像的典型噪声模型."""
    rng = np.random.default_rng(seed)
    s = sigma / 255.0
    real = image + rng.normal(0, s, image.shape)
    imag = rng.normal(0, s, image.shape)
    return np.clip(np.sqrt(real ** 2 + imag ** 2), 0, 1)


def add_speckle_noise(image: np.ndarray, sigma: float = 15.0,
                      seed: int = 0) -> np.ndarray:
    """斑点噪声: 超声图像的典型乘性噪声模型."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma / 100.0, image.shape)
    return np.clip(image + image * noise, 0, 1)


NOISE_MODELS = {
    "gaussian": add_gaussian_noise,
    "rician": add_rician_noise,
    "speckle": add_speckle_noise,
}
