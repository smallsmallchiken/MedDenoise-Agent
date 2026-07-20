"""MedDenoise-Agent 命令行入口.

用法:
  python main.py demo                       # 内置CT体模演示(合成噪声+全流程)
  python main.py process <图像路径> [-o 输出目录] [--sigma 15]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage import io

from meddenoise import data as med_data
from meddenoise.agent.coordinator import Coordinator
from meddenoise.report import generate_report


def save_outputs(result: dict, noisy, reference, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    io.imsave(out_dir / f"{name}_denoised.png",
              (result["output"] * 255).astype("uint8"))
    (out_dir / f"{name}_report.md").write_text(
        generate_report(result, name), encoding="utf-8")

    cols = [("噪声图像", noisy), ("Agent处理结果", result["output"])]
    if reference is not None:
        cols.insert(0, ("原始图像", reference))
    fig, axes = plt.subplots(1, len(cols), figsize=(4 * len(cols), 4))
    plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "SimHei",
                                       "Noto Sans CJK SC", "DejaVu Sans"]
    for ax, (title, img) in zip(axes, cols):
        ax.imshow(img, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_comparison.png", dpi=150)
    print(f"结果已保存至 {out_dir}/")


def cmd_demo(args):
    images = med_data.load_sample_images()
    reference = images["ct_phantom"]
    noisy = med_data.add_gaussian_noise(reference, sigma=args.sigma)
    coordinator = Coordinator()
    result = coordinator.process(noisy, reference, "ct_phantom.png")
    save_outputs(result, noisy, reference, Path(args.output), "demo")


def cmd_process(args):
    noisy = med_data.load_image(args.image)
    coordinator = Coordinator()
    result = coordinator.process(noisy, None, Path(args.image).name)
    save_outputs(result, noisy, None, Path(args.output), Path(args.image).stem)


def main():
    parser = argparse.ArgumentParser(description="MedDenoise-Agent 医学影像智能去噪系统")
    sub = parser.add_subparsers(dest="command", required=True)

    p_demo = sub.add_parser("demo", help="内置演示")
    p_demo.add_argument("--sigma", type=float, default=25.0)
    p_demo.add_argument("-o", "--output", default="outputs")
    p_demo.set_defaults(func=cmd_demo)

    p_proc = sub.add_parser("process", help="处理指定图像")
    p_proc.add_argument("image")
    p_proc.add_argument("-o", "--output", default="outputs")
    p_proc.set_defaults(func=cmd_process)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
