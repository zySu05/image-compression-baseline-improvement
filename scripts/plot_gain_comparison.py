import csv
from pathlib import Path
import matplotlib.pyplot as plt

old_path = Path.home() / "roi-compression/results/hyperprior_rd_summary.csv"
new_path = Path("results/gain/gain_kodak_summary.csv")
out_path = Path("results/gain/gain_vs_baseline.png")

def read_csv(path):
    with path.open("r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

old = read_csv(old_path)
new = read_csv(new_path)

assert len(old) == 4
assert len(new) == 5

def numbers(rows, x, y):
    pts = sorted((float(r[x]), float(r[y])) for r in rows)
    return list(zip(*pts))

bx, by = numbers(old, "bpp", "psnr_db")
gx, gy = numbers(new, "avg_bpp", "avg_psnr_db")

plt.figure(figsize=(8, 5.5))
plt.plot(bx, by, "o-", label="Hyperprior Q1-Q4 (multi-model)")
plt.plot(gx, gy, "s-", label="Q1 + Gain (single-model)")

for row in old:
    plt.annotate(
        f"Q{row['quality']}",
        (float(row["bpp"]), float(row["psnr_db"])),
        xytext=(4, 8),
        textcoords="offset points",
    )

for row in new:
    plt.annotate(
        f"G={row['gain']}",
        (float(row["avg_bpp"]), float(row["avg_psnr_db"])),
        xytext=(4, -13),
        textcoords="offset points",
    )

plt.xlabel("Bitrate (bpp)")
plt.ylabel("PSNR (dB)")
plt.title("Kodak: Multi-model Hyperprior vs Single-model Gain")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig(out_path, dpi=200)
plt.close()

print("对比曲线生成成功：", out_path)
