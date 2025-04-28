from __future__ import annotations

import time
from pathlib import Path

from jetsontools import Tegrastats, get_powerdraw, parse_tegrastats, filter_data


example_path = Path("tegra-out.out")

interval = 5  # sample every 5 ms
duration = 160  # 160 seconds of sampling
timestamps: list[tuple[float, float]] = []

t0 = time.time()
with Tegrastats(example_path, interval):
    t00 = time.time()
    time.sleep(duration)  # REPLACE THIS WITH YOUR WORK UNIT
    t11 = time.time()
    timestamps.append((t00, t11))
t1 = time.time()
total = t1 - t0

print(f"Execution took: {round(total, 3)} for 5 seconds measured.")
print("This is due to waiting for tegrastats process to open.")

# should be roughly 1000 / interval * duration entries
with example_path.open("r") as f:
    lines = f.readlines()

    print(
        f"Total of: {len(lines)} entries found, compared to {1000 / interval * duration}.",
    )
    print("Loss is expected.")

# parse the output
output = parse_tegrastats(example_path)

# filter the data to only include values inside the timestamps
filtered, _ = filter_data(output, timestamps)

# parse the energy
energy_data = get_powerdraw(filtered)

with open("power.csv", "w") as f:
    f.write('\n'.join(str(v) for v in energy_data["VDD_TOTAL"].raw))

# for mname, metric in energy_data.items():
#     print(f"{mname}: {metric.mean} mW")