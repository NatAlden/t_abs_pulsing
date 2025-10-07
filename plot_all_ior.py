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

    ax_main.plot(-df["depth"], df["refractive_index"], color='violet', linestyle='-.', label= 'greenland 3-part exponential model', lw = 2)
    ax_res.plot(-df["depth"], np.zeros(len(df["depth"])), color='violet', linestyle='-.', lw = 2)

    for station, drop_name, color, ls in zip(stations, drop_names, colors, linestyles):
        [depths, ior] = np.load(f"processed_data/{station}_{drop_name}.npy")
        dfdata = pd.DataFrame({"depth" : depths, "ior" : ior})
        dfdata.to_csv(f"processed_data/{station}_{drop_name}.csv", index=False)

        hole_names = drop_name.split('_')[1:]

        model_interp = np.interp(depths, df["depth"], df["refractive_index"])
        ax_main.plot(-depths, ior, linestyle=ls, color = color, label= f'station {station}: Pulsing {hole_names[0]} to {hole_names[1]}')#', baseline {baseline:.2f}m)')
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
    stations = [34, 34, 35, 35, 25, 25]
    drop_names = ["pulsing_b_a_down", "pulsing_b_c_down", "pulsing_a_b_up", "pulsing_a_c_up",
        "pulsing_a_b", "pulsing_a_c_up"]
    linestyles = ["-", "--", "-", "--", "-", "--"]
    colors = ["blue", "blue", "#FCD116", "#FCD116", "#CE1126", "#CE1126"]

    plot_multiple_measurements(stations, drop_names, linestyles, colors)
    