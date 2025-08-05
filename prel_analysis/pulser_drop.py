
import pandas as pd
import matplotlib.pyplot as plt
import imageio
import os
from glob import glob
from scipy.signal import find_peaks
import numpy as np
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('stat', type=str, help='station_XX')
parser.add_argument('meas', type=str, help='pulsing_a_b')

args = parser.parse_args()

script_dir = os.path.dirname(__file__)

data_folder = os.path.join(script_dir, f"../raw_data/{args.stat}/{args.meas}/")
metadata_file = os.path.join(script_dir, f"../raw_data/{args.stat}/meta_{args.meas}.csv")
save_plots = os.path.join(script_dir, f"../prel_analysis/results/{args.stat}/{args.meas}/plots/")

with open(metadata_file, 'r') as f:
    lines = f.readlines()

for idx, line in enumerate(lines):
    if line.strip().startswith("#hole A [m]"):
        data_start = idx
        break

metadata = pd.read_csv(metadata_file, skiprows=data_start, sep=None, engine="python")
metadata.columns = ["hole_a_m", "hole_b_m", "filename", "alt_filename"]
metadata = metadata.dropna(subset=["filename"])

#first_file = metadata.iloc[0]["filename"]
#data_file = os.path.join(data_folder, first_file + ".CSV")

column_map = {
    'time': 'in s',
    'ch1': 'C1 in V',
    'ch3': 'C3 in V'
}

gif_frames = []

for _, row in metadata.iterrows():
    fname = row["filename"]
    hole_depth = row["hole_a_m"]
    file_path = os.path.join(data_folder, fname + ".CSV")

    print(fname)

    if not os.path.isfile(file_path):
        print(f"⚠️ Missing: {fname}.CSV")
        continue

    try:
        df = pd.read_csv(file_path)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df[column_map['time']], df[column_map['ch3']], color='blue')
        ax.set_title(f"depth = {np.round(float(hole_depth), 1)} m")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("ch3 (V)")
        ax.set_ylim(-0.005, 0.005)
        ax.set_xlim(0.7e-6, 1e-6)
        ax.grid(True)
        plt.tight_layout()

        frame_path = "frame.png"
        plt.savefig(frame_path)
        gif_frames.append(imageio.v2.imread(frame_path))
        plt.close()

    except Exception as e:
        print(f"⚠️ Skipping {fname}: {e}")


imageio.mimsave(save_plots + "time_vs_ch3.gif", gif_frames, fps=4)