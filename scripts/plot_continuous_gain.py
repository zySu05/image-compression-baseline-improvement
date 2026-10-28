import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

def plot(rows, xcol, ycol, name, **kwargs):
    rows = sorted(rows, key=lambda r: float(r[xcol]))
    plt.plot(
        [float(r[xcol]) for r in rows],
        [float(r[ycol]) for r in rows],
        marker="o",
        label=name,
        **kwargs,
    )

continuous = read_csv(
    "results/continuous/continuous_kodak_summary.csv"
)

plt.figure(figsize=(9, 6))

plot(
    continuous, "avg_bpp", "avg_psnr_db",
    "Continuous Gain (8-step trained)"
)

fixed = Path("results/gain/gain_kodak_summary.csv")
if fixed.exists():
    plot(
        read_csv(fixed),
        "avg_bpp", "avg_psnr_db",
        "Fixed scalar Gain",
        linestyle="--",
    )

baseline = Path.home() / (
    "roi-compression/results/hyperprior_rd_summary.csv"
)
if baseline.exists():
    rows = read_csv(baseline)
    print("Hyperprior baseline columns:", list(rows[0]))
    # 不猜测 CSV 字段名：字段不匹配时提示检查
    x_candidates = ["avg_bpp", "bpp"]
    y_candidates = ["avg_psnr_db", "psnr_db", "psnr"]
    xcol = next((x for x in x_candidates if x in rows[0]), None)
    ycol = next((y for y in y_candidates if y in rows[0]), None)

    if xcol and ycol:
        plot(rows, xcol, ycol, "Multi-model Hyperprior")
    else:
        print("基线CSV字段未匹配，暂不绘制多模型曲线")

plt.xlabel("Bits per pixel (bpp)")
plt.ylabel("PSNR (dB)")
plt.title("Kodak-24 Rate-Distortion Comparison")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

out = Path("results/continuous/continuous_rd_comparison.png")
plt.savefig(out, dpi=180)
print("RD曲线已保存:", out)
