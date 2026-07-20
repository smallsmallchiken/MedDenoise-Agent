"""下载DnCNN预训练权重 (KAIR模型库 dncnn_25.pth, σ=25 灰度模型).

运行: python scripts/download_weights.py
"""

import sys
import urllib.request
from pathlib import Path

URL = "https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_25.pth"
DEST = Path(__file__).resolve().parents[1] / "weights" / "dncnn_25.pth"


def main():
    if DEST.exists():
        print(f"权重已存在: {DEST}")
        return
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"下载 {URL} ...")
    urllib.request.urlretrieve(URL, DEST)
    print(f"已保存至 {DEST} ({DEST.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
