import pandas as pd
import matplotlib.pyplot as plt
import imageio
import os
from glob import glob
from scipy.signal import find_peaks
import numpy as np
from scipy.constants import c
import argparse
from geopy.distance import geodesic

import sys
sys.path.append('/Users/nathanielalden/Downloads/RNO-G/t_abs_pulsing')
import delay_utils

def load_GPS_distance(station, drop_name):
    #TODO: load in yaml file when ready
    #if drop_name = 
    if drop_name == "pulsing_b_a_down":
        point_a = (72.617925736, -38.410023642)
        point_b = (72.618200468, -38.409498297)
    elif drop_name == "pulsing_b_c_down":
        point_a = (72.617925736, -38.410023642)
        point_b = (72.618196808, -38.410535718)

    baseline = geodesic(point_a, point_b).meters
    print(f"Distance: {baseline:.4f} meters for station {station}, {drop_name}")

    return baseline

def find_cable_delays(station, drop_name):
    script_dir = os.path.dirname(__file__)
    TDR_folder = f"{script_dir}/raw_data/station_{station}/cableTDR_{drop_name}"
    if drop_name == "pulsing_b_a_down":
        rx_file = "SRF_T73.CSV"
        tx_file = "SRF_T74.CSV"
    elif drop_name == "pulsing_b_c_down":
        rx_file = "SRF_T114.CSV"
        tx_file = "SRF_T115.CSV"

    cable_a = delay_utils.get_cable_delay(f"{TDR_folder}/{rx_file}", cable = "TX")*1e-9
    cable_b = delay_utils.get_cable_delay(f"{TDR_folder}/{tx_file}", cable = "RX")*1e-9

    #cable_a = 557.32e-9
    #cable_b = 307.38e-9

    jumper_cable = 2.968e-9

    return cable_a, cable_b, jumper_cable

def find_time_delay(time, ch1, ch3, thresh = 0.001):
    min_ch1_idx = ch1.argmin()
    min_ch1_time = time[min_ch1_idx]

    above_thresh = (ch3 > thresh).nonzero()[0]
    if len(above_thresh) == 0:
        raise ValueError(f"ch3 never exceeds {thresh} V")

    start_idx = above_thresh[0]
    peaks, _ = find_peaks(ch3[start_idx:])
    if len(peaks) == 0:
        raise ValueError("No peaks after threshold")

    peak_time = time[start_idx + peaks[0]]
    delta_t = peak_time - min_ch1_time

    return delta_t


def find_time_and_depth(data_folder, metadata_file):
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

            delta_t = find_time_delay(time, ch1, ch3)

            depths.append(hole_a_depth)
            delta_t_list.append(delta_t)

        except Exception as e:
            print(f"⚠️ Skipping {fname}: {e}")

    return np.array(depths), np.array(delta_t_list)

def plot_multiple_measurements(stations, drop_names):
    script_dir = os.path.dirname(__file__)
    save_plots = os.path.join(script_dir, f"plots/")
    #save_plots = os.path.join(script_dir, f"{rootdir}/prel_analysis/results/{args.station}/{args.meas}/plots/")

    plt.figure(figsize=(10, 6))
    for station, drop_name in zip(stations, drop_names):
        print(station, drop_name)
        data_folder = os.path.join(script_dir, f"raw_data/station_{station}/{drop_name}/")
        metadata_file = os.path.join(script_dir, f"raw_data/station_{station}/meta_{drop_name}.csv")
        
        baseline = load_GPS_distance(station, drop_name)
        cable_a, cable_b, jumper_cable = find_cable_delays(station, drop_name)

        depths, delta_t = find_time_and_depth(data_folder, metadata_file)
        ior = (c * (delta_t - cable_a - cable_b + jumper_cable)) / baseline
        plt.plot(-depths, ior, marker='o', linestyle='-', label= f'station {station}: Baseline {baseline:.2f}m)')

    #Now overplot ice models
    three_part_file = os.path.join(script_dir, f"raw_data/Greenland_ice_model.csv")  
    df = pd.read_csv(three_part_file, header=None, names=["depth", "refractive_index"])
    plt.plot(-df["depth"], df["refractive_index"], color='r', linestyle='--', label= 'greenland three part model')

    plt.xlabel("Hole Depth (m)")
    plt.ylabel("Refractive Index")
    plt.grid(True)
    plt.legend()
    plt.gca().invert_xaxis()  # Optional: deeper values lower
    plt.tight_layout()
    #plt.show()
    plt.xlim(right = -100)
    plt.ylim(top = 1.8)
    plt.savefig(save_plots + 'refractive_index.png')

if __name__ == "__main__":
    stations = [34, 34]
    drop_names = ["pulsing_b_a_down", "pulsing_b_c_down"]

    plot_multiple_measurements(stations, drop_names)
