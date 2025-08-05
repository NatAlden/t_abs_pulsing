import pandas as pd
import matplotlib.pyplot as plt
import imageio
import os
from glob import glob
from scipy.signal import find_peaks
import numpy as np


# Path where the CSV files are stored
data_folder_1 = "/Users/nilhe916/PhD_Uppsala/2025/Greenland/010825/cableTDR/SRF_T122_A.CSV"  # Change if needed


df = pd.read_csv(data_folder_1)


df.columns = df.columns.str.strip()  # Strip any extra spaces

# Time threshold
threshold_time = 0.5e-6

# Split the data
df_before = df[df["in s"] < threshold_time]
df_after = df[df["in s"] >= threshold_time]

# Find minima in ch3
idx_min_before = df_before["C3 in V"].idxmin()
idx_min_after = df_after["C3 in V"].idxmin()

# Get corresponding times
t_min_before = df.loc[idx_min_before, "in s"]
t_min_after = df.loc[idx_min_after, "in s"]

# Time delay
delta_t = t_min_after - t_min_before

# Print results
print(f"Minimum before 0.5 µs at {t_min_before:.2e} s")
print(f"Minimum after 0.5 µs at {t_min_after:.2e} s")
print(f"⏱️ Time delay cable A: {delta_t:.2e} seconds")


# ✅ Plot C1 and C3 vs time
plt.figure(figsize=(8, 4))
plt.plot(df["in s"], df["C1 in V"], label="Channel 1 (C1)", color="orange")
plt.plot(df["in s"], df["C3 in V"], label="Channel 3 (C3)", color="blue")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.title("Waveform: C1 and C3 vs Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig('cable_d_A.png')



data_folder_2 = "/Users/nilhe916/PhD_Uppsala/2025/Greenland/010825/cableTDR/SRF_T123_B.CSV"  # Change if needed


df = pd.read_csv(data_folder_2)


df.columns = df.columns.str.strip()  # Strip any extra spaces

# Time threshold
threshold_time = 0.5e-6

# Split the data
df_before = df[df["in s"] < threshold_time]
df_after = df[df["in s"] >= threshold_time]

# Find minima in ch3
idx_min_before = df_before["C3 in V"].idxmin()
idx_min_after = df_after["C3 in V"].idxmin()

# Get corresponding times
t_min_before = df.loc[idx_min_before, "in s"]
t_min_after = df.loc[idx_min_after, "in s"]

# Time delay
delta_t = t_min_after - t_min_before

# Print results
print(f"Minimum before 0.5 µs at {t_min_before:.2e} s")
print(f"Minimum after 0.5 µs at {t_min_after:.2e} s")
print(f"⏱️ Time delay cable B: {delta_t:.2e} seconds")

# ✅ Plot C1 and C3 vs time
plt.figure(figsize=(8, 4))
plt.plot(df["in s"], df["C1 in V"], label="Channel 1 (C1)", color="orange")
plt.plot(df["in s"], df["C3 in V"], label="Channel 3 (C3)", color="blue")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.title("Waveform: C1 and C3 vs Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig('cable_d_B.png')