
import pandas as pd
import matplotlib.pyplot as plt
import imageio
import os
from glob import glob
from scipy.signal import find_peaks
import numpy as np
from scipy.constants import c
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('stat', type=str, help='station_XX')
parser.add_argument('meas', type=str, help='pulsing_a_b')

parser.add_argument('bound', type=float, nargs='?', default=0.001,)

args = parser.parse_args()

script_dir = os.path.dirname(__file__)

data_folder = os.path.join(script_dir, f"../raw_data/{args.stat}/{args.meas}/")
metadata_file = os.path.join(script_dir, f"../raw_data/{args.stat}/meta_{args.meas}.csv")
save_plots = os.path.join(script_dir, f"../prel_analysis/results/{args.stat}/{args.meas}/plots/")
 
three_part_file = os.path.join(script_dir, f"../raw_data/Greenland_ice_model.csv")  

"""
from geopy.distance import geodesic

point_a = (72.630667604, -38.441826547)
point_b = (72.63094567500001, -38.441286414)

distance_m = geodesic(point_a, point_b).meters
print(f"Distance: {distance_m:.4f} meters")
"""
baseline = 34

cable_a = 307.4e-9
cable_b = 307.4e-9

with open(metadata_file, 'r') as f:
    lines = f.readlines()

for idx, line in enumerate(lines):
    if line.strip().startswith("#hole A [m]"):
        data_start = idx
        break

metadata = pd.read_csv(metadata_file, skiprows=data_start + 1, sep=None, engine="python")
metadata.columns = ["hole_a_m", "hole_b_m", "filename", "alt_filename"]
metadata = metadata.dropna(subset=["filename"])  

depths = []
refractive_indices = []
delta_t_list = []

for i, row in metadata.iterrows():
    fname = row["filename"]
    hole_a_depth = row["hole_a_m"]
    full_path = os.path.join(data_folder, fname + ".CSV")

    if not os.path.isfile(full_path):
        print(f"⚠️ File not found: {fname}.CSV")
        continue

    try:
        df = pd.read_csv(full_path)
        column_map = {
            'time': 'in s',
            'ch1': 'C1 in V',
            'ch3': 'C3 in V'
        }

        time = df[column_map['time']].values
        ch1 = df[column_map['ch1']].values
        ch3 = df[column_map['ch3']].values

        min_ch1_idx = ch1.argmin()
        min_ch1_time = time[min_ch1_idx]

        above_thresh = (ch3 > args.bound).nonzero()[0]
        if len(above_thresh) == 0:
            raise ValueError(f"ch3 never exceeds {args.bound} V")

        start_idx = above_thresh[0]
        peaks, _ = find_peaks(ch3[start_idx:])
        if len(peaks) == 0:
            raise ValueError("No peaks after threshold")

        peak_time = time[start_idx + peaks[0]]
        delta_t = peak_time - min_ch1_time

        n = (c * (delta_t - cable_a - cable_b)) / baseline

        depths.append(hole_a_depth)
        refractive_indices.append(n)
        delta_t_list.append(1e9*delta_t)

    except Exception as e:
        print(f"⚠️ Skipping {fname}: {e}")


z_bottom = -3000
n_ice = 1.78
z_0 = 37.25
delta_n = 0.51
z_shift = 0
all_depth = -np.linspace(0, 40, 1000)
n_g_simple = n_ice - delta_n * np.exp((all_depth - z_shift) / z_0)


df = pd.read_csv(three_part_file, header=None, names=["depth", "refractive_index"])


plt.plot(df["depth"], df["refractive_index"], marker='o', linestyle='-')

results_index = np.array([- np.array(depths), refractive_indices])
np.save(save_plots + 'refractive_index.npy', results_index)

plt.figure(figsize=(10, 6))
plt.plot(- np.array(depths), refractive_indices, marker='o', linestyle='-', label= f'2025 data (hole distance {baseline}m)')
plt.plot(all_depth, n_g_simple, color='r', linestyle='-', label= 'greenland simple model')
plt.plot(-df["depth"], df["refractive_index"], color='r', linestyle='--', label= 'greenland three part model')

#plt.plot(three_part_n["depth_3"], three_part_n["index_3"], color='r', linestyle='--', label= 'greenland three part model')
#plt.plot(depths, delta_t, marker='o', linestyle='-')
plt.xlabel("Hole A Depth (m)")
plt.ylabel("Refractive Index")
plt.title("Refractive Index vs Hole A Depth")
plt.grid(True)
plt.legend()
plt.gca().invert_xaxis()  # Optional: deeper values lower
plt.tight_layout()
#plt.show()
plt.savefig(save_plots + 'refractive_index.png')

plt.figure(figsize=(10, 6))
#plt.plot(depths, refractive_indices, marker='o', linestyle='-')
plt.plot(depths, delta_t_list, marker='o', linestyle='-')
plt.xlabel("Hole A Depth (m)")
plt.ylabel("time difference [ns]")
plt.title("Refractive Index vs Hole A Depth")
plt.grid(True)
plt.gca().invert_xaxis()  # Optional: deeper values lower
plt.tight_layout()
plt.savefig(save_plots + 'time_delays.png')