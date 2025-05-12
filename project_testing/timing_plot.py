import pandas as pd
import matplotlib.pyplot as plt
import os


# 0) helper to parse either "M:SS.xxx" → seconds or pass through floats
def parse_min_sec(x):
    s = str(x)
    if ":" not in s:
        return float(s)
    mins, secs = s.split(":", 1)
    return int(mins) * 60 + float(secs)


# 1) Read in your CSV
csv_path = "timing_results.csv"
df = pd.read_csv(csv_path)

# 2) Normalize the Datapoints column ("10k" → 10000)
df["Datapoints"] = df["Datapoints"].str.rstrip("k").astype(int) * 1000

# 3) Ensure trial columns are strings, then parse each into seconds
time_cols = ["Trial_1", "Trial_2", "Trial_3"]
df[time_cols] = df[time_cols].astype(str)
for c in time_cols:
    df[c] = df[c].apply(parse_min_sec)

# 4) Compute the average time per row
df["AvgTime_s"] = df[time_cols].mean(axis=1)

# 5) Pivot so index=Datapoints, columns=Application, values=AvgTime_s
pivot = df.pivot(index="Datapoints", columns="Application", values="AvgTime_s")
pivot.columns = [app.replace("_", " on ").upper() for app in pivot.columns]

# 6) Plot grouped bar chart
ax = pivot.plot(kind="bar", figsize=(10, 6), width=0.8, edgecolor="black")

# 7) Annotate each bar with its height (rounded to 2 decimals)
for bar in ax.patches:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,  # x position = center of bar
        height,  # y position = top of bar
        f"{height:.2f}",  # label = height
        ha="center",
        va="bottom",
        fontsize=10,
    )

# 8) Final formatting
ax.set_xlabel("Number of datapoints", fontsize=12)
ax.set_ylabel("Average time (s)", fontsize=12)
ax.set_title("Average Run Time by Model & Hardware", fontsize=14, fontweight="bold")
ax.set_xticklabels([f"{int(x):,}" for x in pivot.index], rotation=0)
ax.legend(title="Configuration", bbox_to_anchor=(1.02, 1), loc="upper left")
ax.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()

# 9) Save & show
out_png = os.path.join(os.path.dirname(csv_path), "timing_grouped.png")
plt.savefig(out_png, dpi=300)
plt.show()
