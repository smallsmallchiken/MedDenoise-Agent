"""对比实验: 各去噪算法在多噪声水平下的PSNR/SSIM定量评测.

生成论文所需的实验表格(Markdown/CSV)与可视化对比图.
运行: python experiments/run_experiments.py
"""

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from meddenoise import data as med_data
from meddenoise.tools import denoise, metrics

plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "SimHei",
                                   "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = Path(__file__).resolve().parent / "results"
METHODS = ["gaussian", "median", "wavelet", "nlm", "bm3d", "sa-bm3d"]
SIGMAS = [10, 15, 25, 35]
IMAGE_NAME = "ct_phantom"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    reference = med_data.load_sample_images()[IMAGE_NAME]

    rows = []
    visuals = {}
    for sigma in SIGMAS:
        noisy = med_data.add_gaussian_noise(reference, sigma=sigma)
        noisy_metrics = metrics.full_reference_metrics(reference, noisy)
        rows.append({"sigma": sigma, "method": "noisy",
                     "time_s": 0.0, **noisy_metrics})
        print(f"\n=== σ={sigma} 噪声图像 PSNR={noisy_metrics['psnr']} "
              f"SSIM={noisy_metrics['ssim']} ===")
        for method in METHODS:
            start = time.time()
            output = denoise.denoise(noisy, method, sigma)
            elapsed = time.time() - start
            m = metrics.full_reference_metrics(reference, output)
            rows.append({"sigma": sigma, "method": method,
                         "time_s": round(elapsed, 2), **m})
            print(f"  {method:10s} PSNR={m['psnr']:6.2f} SSIM={m['ssim']:.4f} "
                  f"({elapsed:.1f}s)")
            if sigma == 25:
                visuals[method] = output
        if sigma == 25:
            visuals["noisy"] = noisy

    with open(OUT_DIR / "benchmark.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = ["# 去噪算法对比实验结果", "",
             f"测试图像: Shepp-Logan CT体模 (256x256), 高斯噪声", "",
             "| σ | 方法 | PSNR(dB) | SSIM | RMSE | 耗时(s) |",
             "| --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        lines.append(f"| {r['sigma']} | {r['method']} | {r['psnr']} "
                     f"| {r['ssim']} | {r['rmse']} | {r['time_s']} |")
    (OUT_DIR / "benchmark.md").write_text("\n".join(lines), encoding="utf-8")

    order = ["noisy"] + METHODS
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    axes[0].imshow(reference, cmap="gray")
    axes[0].set_title("原始图像")
    for ax, name in zip(axes[1:], order):
        ax.imshow(visuals[name], cmap="gray")
        m = next(r for r in rows if r["sigma"] == 25 and r["method"] == name)
        label = "噪声图像 σ=25" if name == "noisy" else name.upper()
        ax.set_title(f"{label}\nPSNR={m['psnr']} SSIM={m['ssim']}")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "comparison_sigma25.png", dpi=150)

    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for method in METHODS:
        pts = [(r["sigma"], r["psnr"], r["ssim"]) for r in rows
               if r["method"] == method]
        ax1.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=method)
        ax2.plot([p[0] for p in pts], [p[2] for p in pts], marker="o", label=method)
    ax1.set_xlabel("噪声强度 σ"); ax1.set_ylabel("PSNR (dB)"); ax1.set_title("PSNR对比"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.set_xlabel("噪声强度 σ"); ax2.set_ylabel("SSIM"); ax2.set_title("SSIM对比"); ax2.legend(); ax2.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(OUT_DIR / "metrics_curves.png", dpi=150)
    print(f"\n实验结果已保存至 {OUT_DIR}/")


if __name__ == "__main__":
    main()
