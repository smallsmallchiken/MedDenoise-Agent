"""对比实验: 各去噪算法在Set12与BSD300数据集上的PSNR/SSIM定量评测.

生成论文所需的实验表格(Markdown/CSV)与可视化对比图.
运行: python experiments/run_experiments.py [--quick]
"""

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from meddenoise import data as med_data
from meddenoise.tools import denoise, metrics

plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "SimHei",
                                   "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent / "results"
SET12_DIR = ROOT / "data" / "datasets" / "Set12"
BSD300_DIR = ROOT / "data" / "datasets" / "BSDS300" / "images" / "test"
METHODS = ["gaussian", "median", "wavelet", "nlm", "bm3d", "vt-bm3d", "dncnn"]
SIGMAS = [15, 25, 35]
VISUAL_IMAGE = "Set12/05"  # σ=25 视觉对比用图


def load_dataset_images(quick: bool = False) -> dict[str, np.ndarray]:
    images: dict[str, np.ndarray] = {}
    set12 = sorted(SET12_DIR.glob("*.png"))[: 4 if quick else 12]
    for f in set12:
        images[f"Set12/{f.stem}"] = med_data.load_image(str(f))
    if BSD300_DIR.exists():
        bsd = sorted(BSD300_DIR.glob("*.jpg"))[: 2 if quick else 8]
        for f in bsd:
            images[f"BSD300/{f.stem}"] = med_data.load_image(str(f), size=256)
    return images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="快速模式(少量图片)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    images = load_dataset_images(args.quick)
    print(f"测试图像数量: {len(images)}")

    rows = []
    visuals: dict[str, np.ndarray] = {}
    for sigma in SIGMAS:
        agg: dict[str, list] = {m: [] for m in ["noisy"] + METHODS}
        times: dict[str, list] = {m: [] for m in METHODS}
        for name, reference in images.items():
            noisy = med_data.add_gaussian_noise(reference, sigma=sigma)
            agg["noisy"].append((metrics.psnr(reference, noisy),
                                 metrics.ssim(reference, noisy)))
            for method in METHODS:
                start = time.time()
                output = denoise.denoise(noisy, method, sigma)
                times[method].append(time.time() - start)
                agg[method].append((metrics.psnr(reference, output),
                                    metrics.ssim(reference, output)))
                if sigma == 25 and name == VISUAL_IMAGE:
                    visuals[method] = output
            if sigma == 25 and name == VISUAL_IMAGE:
                visuals["noisy"] = noisy
                visuals["reference"] = reference

        print(f"\n=== σ={sigma} (平均于{len(images)}张图) ===")
        for method in ["noisy"] + METHODS:
            psnr_avg = float(np.mean([p for p, _ in agg[method]]))
            ssim_avg = float(np.mean([s for _, s in agg[method]]))
            time_avg = float(np.mean(times[method])) if method in times else 0.0
            rows.append({"sigma": sigma, "method": method,
                         "psnr": round(psnr_avg, 2), "ssim": round(ssim_avg, 4),
                         "time_s": round(time_avg, 2)})
            print(f"  {method:10s} PSNR={psnr_avg:6.2f} SSIM={ssim_avg:.4f} "
                  f"({time_avg:.2f}s)")

    with open(OUT_DIR / "benchmark.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = ["# 去噪算法对比实验结果", "",
             f"测试数据: Set12 + BSD300(test) 共{len(images)}张灰度图, "
             "高斯噪声, 指标为数据集平均值", "",
             "| σ | 方法 | PSNR(dB) | SSIM | 平均耗时(s) |",
             "| --- | --- | --- | --- | --- |"]
    for r in rows:
        lines.append(f"| {r['sigma']} | {r['method']} | {r['psnr']} "
                     f"| {r['ssim']} | {r['time_s']} |")
    (OUT_DIR / "benchmark.md").write_text("\n".join(lines), encoding="utf-8")

    if visuals:
        order = ["reference", "noisy"] + METHODS
        fig, axes = plt.subplots(3, 3, figsize=(13, 13))
        axes = axes.ravel()
        m25 = {r["method"]: r for r in rows if r["sigma"] == 25}
        for ax, name in zip(axes, order):
            ax.imshow(visuals[name], cmap="gray")
            if name == "reference":
                ax.set_title("原始图像")
            elif name == "noisy":
                ax.set_title(f"噪声图像 σ=25\nPSNR={m25['noisy']['psnr']}")
            else:
                ax.set_title(f"{name.upper()}\nPSNR={m25[name]['psnr']} "
                             f"SSIM={m25[name]['ssim']}")
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
