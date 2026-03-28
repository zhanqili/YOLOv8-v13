import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 读取 YOLOv13 的 results.csv
csv_path = "runs/detect/yolo13_res/results.csv"
df = pd.read_csv(csv_path)

# 去掉列名前后的空格
df.columns = df.columns.str.strip()

# 把字符串转成数值，无法转换的设为 NaN
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 把 inf 和 -inf 替换掉，避免图像异常
df = df.replace([np.inf, -np.inf], np.nan)

# 定义要画的列（适配 YOLOv9）
plot_items = [
     "train/box_loss",
    "train/cls_loss",
    "train/dfl_loss",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "val/box_loss",
    "val/cls_loss",
    "val/dfl_loss",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)"
]

fig, axes = plt.subplots(2, 5, figsize=(18, 8))
axes = axes.ravel()

for i, col in enumerate(plot_items):
    ax = axes[i]

    if col in df.columns:
        y = df[col]

        # 平滑曲线
        smooth = y.rolling(window=5, min_periods=1).mean()

        ax.plot(y, marker='o', linewidth=2, label='results')
        ax.plot(smooth, linestyle=':', linewidth=2, label='smooth')
        ax.set_title(col)
        ax.legend()
    else:
        ax.set_title(f"{col}\n(not found)")
        ax.axis("off")

plt.tight_layout()

# 保存路径
save_path = "runs/detect/yolo13_res"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"图像已保存到: {save_path}")