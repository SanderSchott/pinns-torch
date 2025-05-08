import os
import pandas as pd
import matplotlib.pyplot as plt

directory     = "/home/sschott/CSCI582-Final-Project/pinns-torch/project_testing"
time_interval = 0.005  # 5 ms

# 1) Build a long DataFrame of all runs
rows = []
for fn in os.listdir(directory):
    if not fn.endswith("_power_averaged.csv"):
        continue
    # filename format: 10k_dnn_gpu_power_averaged.csv
    base, _, _ = fn.partition("_power_averaged.csv")
    num_data, app, hw = base.split('_')      # e.g. ["10k","dnn","gpu"]
    dp = int(num_data.rstrip('k')) * 1000    # 10k → 10000

    df = pd.read_csv(os.path.join(directory, fn))
    power = df.iloc[:,0].astype(float)
    energy = power.sum() * time_interval     # mJ

    rows.append({
        "datapoints": dp,
        "app":        app.upper(),            # DNN / PINN
        "hw":         hw.upper(),             # GPU / CPU / MPS
        "energy":     energy
    })

df = pd.DataFrame(rows)
df.sort_values("datapoints", inplace=True)

# 2) Pivot: index=datapoints, columns=(app,hw), values=energy
pivot = df.pivot(index="datapoints", columns=["app","hw"], values="energy")

# (optional) flatten the column MultiIndex to nice labels
pivot.columns = [f"{app} on {hw}" for app,hw in pivot.columns]

# 3) Plot grouped bars
ax = pivot.plot(
    kind="bar",
    figsize=(12, 6),
    edgecolor="black",
    width=0.8
)
ax.set_xlabel("Number of Datapoints", fontsize=14)
ax.set_ylabel("Energy (mJ)", fontsize=14)
ax.set_title("Energy Usage Across Different Configurations", fontsize=16, fontweight="bold")
ax.set_xticklabels([f"{int(x):,}" for x in pivot.index], rotation=0, fontsize=12)
ax.legend(title="Run", fontsize=10, title_fontsize=11, loc="upper left")
ax.grid(axis="y", linestyle="--", alpha=0.7)

# Add labels to the top of each bar
for container in ax.containers:
    ax.bar_label(container, fmt="%.1f", label_type="edge", fontsize=10, padding=3)

plt.tight_layout()
plt.savefig(os.path.join(directory, "energy_grouped.png"), dpi=300)
