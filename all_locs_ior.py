import pandas as pd
import matplotlib.pyplot as plt
import imageio
import os
from glob import glob
from scipy.signal import find_peaks, hilbert
import numpy as np
from scipy.constants import c
import argparse
from geopy.distance import geodesic
import yaml
import matplotlib.gridspec as gridspec

import sys
sys.path.append('/Users/nathanielalden/Downloads/RNO-G/t_abs_pulsing')
import delay_utils

def load_yaml(station, yaml_file = "raw_data/geometry.yaml"):
    with open(yaml_file) as stream:
        try:
            all_geometry = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return all_geometry[f"station_{station}"]

def load_GPS_distance(station, drop_name):
    #TODO: load in yaml file when ready
    #if drop_name = 
    station_data = load_yaml(station)
    hole_names = drop_name.split('_')[1:3]
    point_a = (station_data[f"hole_{hole_names[0]}_latitude"], station_data[f"hole_{hole_names[0]}_longitude"])
    point_b = (station_data[f"hole_{hole_names[1]}_latitude"], station_data[f"hole_{hole_names[1]}_longitude"])

    baseline = geodesic(point_a, point_b).meters
    print(f"Distance: {baseline:.4f} meters for station {station}, {drop_name}")

    return baseline

def find_cable_delays(station, drop_name):
    script_dir = os.path.dirname(__file__)
    TDR_folder = f"{script_dir}/raw_data/station_{station}/cableTDR_{drop_name}"
    station_data = load_yaml(station)

    rx_file = station_data['measurements'][drop_name]["cable_a_file"]
    tx_file = station_data['measurements'][drop_name]["cable_b_file"]

    cable_a = delay_utils.get_cable_delay(f"{TDR_folder}/{rx_file}", cable = "TX")*1e-9
    cable_b = delay_utils.get_cable_delay(f"{TDR_folder}/{tx_file}", cable = "RX")*1e-9

    #cable_a = 557.32e-9
    #cable_b = 307.38e-9

    jumper_cable = 2.968e-9

    return cable_a, cable_b, jumper_cable

def find_time_from_template(tvals, sigvals, template_file = "raw_data/pulse_template.npy"):
    tvals, template = np.load(template_file)
    time_from_template_peak = delay_utils.find_delta_t(tvals, sigvals, template)

    print(time_from_template_peak)
    return time_from_template_peak + tvals[np.argmax(template)]

def find_time_delay(time, ch1, ch3, thresh = 0.0005):
    min_ch1_idx = ch1.argmin()
    min_ch1_time = time[min_ch1_idx]

    envelope = np.abs(hilbert(ch3))

    above_thresh = (ch3 > thresh).nonzero()[0]
    if len(above_thresh) == 0:
        raise ValueError(f"ch3 never exceeds {thresh} V")

    start_idx = above_thresh[0]
    peaks, _ = find_peaks(ch3[start_idx:])
    if len(peaks) == 0:
        raise ValueError("No peaks after threshold")

    #peak_time = find_time_from_template(time, ch3)
    peak_time = time[start_idx + peaks[0]]

    #print(f"old peak time is {peak_time}")

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

def plot_multiple_measurements(stations, drop_names, linestyles, colors):
    script_dir = os.path.dirname(__file__)
    save_plots = os.path.join(script_dir, f"plots/")
    #save_plots = os.path.join(script_dir, f"{rootdir}/prel_analysis/results/{args.station}/{args.meas}/plots/")

    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[3, 1])
    ax_main = fig.add_subplot(gs[0, 0])
    ax_res = fig.add_subplot(gs[1, 0])

    #Now overplot ice models
    three_part_file = os.path.join(script_dir, f"raw_data/Greenland_ice_model.csv")  
    df = pd.read_csv(three_part_file, header=None, names=["depth", "refractive_index"])
    ax_main.plot(-df["depth"], df["refractive_index"], color='violet', linestyle='--', label= 'greenland three part model', lw = 2)
    ax_res.plot(-df["depth"], np.zeros(len(df["depth"])), color='violet', linestyle='--')

    for station, drop_name, color, ls in zip(stations, drop_names, colors, linestyles):
        print(station, drop_name)
        data_folder = os.path.join(script_dir, f"raw_data/station_{station}/{drop_name}/")
        metadata_file = os.path.join(script_dir, f"raw_data/station_{station}/meta_{drop_name}.csv")
        
        baseline = load_GPS_distance(station, drop_name)
        cable_a, cable_b, jumper_cable = find_cable_delays(station, drop_name)

        depths, delta_t = find_time_and_depth(data_folder, metadata_file)

        delta_t -= 0.3 / c  #Subtract the in-air time the pulse spends in the borehole (~1 radius on each side)
        baseline -= 0.3   #And correspondingly subtract 2 radii from the baseline

        ior = (c * (delta_t - cable_a - cable_b + jumper_cable)) / baseline
        ax_main.plot(-depths, ior, linestyle=ls, color = color, label= f'station {station}: {drop_name}, baseline {baseline:.2f}m)')
        model_interp = np.interp(depths, df["depth"], df["refractive_index"])
        np.save(f"processed_data/{station}_{drop_name}.npy", [depths, ior])
        ax_res.plot(-depths, ior - model_interp, linestyle=ls, color = color)

    plt.xlabel("Hole Depth (m)")
    ax_main.set_ylabel("Refractive Index")
    ax_main.grid(True)
    ax_main.legend()
    ax_main.invert_xaxis()  # Optional: deeper values lower
    ax_main.set_xlim([0, -100])
    ax_main.set_ylim([1.2,1.8])

    label_x_offset = -0.07
    ax_main.yaxis.set_label_coords(label_x_offset, 0.5)
    ax_res.yaxis.set_label_coords(label_x_offset, 0.5)

    ax_res.set_ylabel("Residual")
    ax_res.set_xlim([0, -100])

    plt.tight_layout()

    plt.savefig(save_plots + 'refractive_index.pdf')

if __name__ == "__main__":
    stations = [34, 34, 35, 35, 35, 35, 25, 25, 25]
    drop_names = ["pulsing_b_a_down", "pulsing_b_c_down", "pulsing_a_b_up", "pulsing_a_c_up", "pulsing_a_b_down", "pulsing_a_b_hpol",
        "pulsing_a_b", "pulsing_a_c_up", "pulsing_a_c_down"]
    linestyles = ["-", "--", "-", "--", ":", "-.", "-", "--", ":"]
    colors = ["blue", "blue", "#FCD116", "#FCD116", "#FCD116", "#FCD116", "#CE1126", "#CE1126", "#CE1126"]

    plot_multiple_measurements(stations, drop_names, linestyles, colors)
    