import pandas as pd
import matplotlib.pyplot as plt
import imageio
import os
from glob import glob
from scipy.signal import find_peaks
import numpy as np
from scipy.constants import c
import argparse

stations = ['station_25', 'station_25', 'station_25', 'station_35', 'station_35', 'station_35', 'station_35']
measurements = ['pulsing_a_b', 'pulsing_a_c_down', 'pulsing_a_c_up', 'pulsing_a_b_down', 'pulsing_a_b_up', 'pulsing_a_b_hpol', 'pulsing_a_c_up']
baseline = [34, 34, 34]

script_dir = os.path.dirname(__file__)
save_plots = os.path.join(script_dir, f"results/")
three_part_file = os.path.join(script_dir, f"../raw_data/Greenland_ice_model.csv")  


plt.figure(figsize=(10, 6))

for idx, meas in enumerate(measurements):

    data = np.load(os.path.join(script_dir, f"../prel_analysis/results/{stations[idx]}/{measurements[idx]}/plots/refractive_index.npy"))
    plt.plot(data[0], data[1], label= f'{stations[idx]}: {measurements[idx]}')

df = pd.read_csv(three_part_file, header=None, names=["depth", "refractive_index"])

plt.plot(-df["depth"], df["refractive_index"], color='r', linestyle='--', label= 'greenland three part model')

plt.xlabel("Hole A Depth (m)")
plt.ylabel("Refractive Index")
plt.title("Refractive Index vs Hole A Depth")
plt.grid(True)
plt.legend()
plt.gca().invert_xaxis()  # Optional: deeper values lower
plt.tight_layout()
#plt.show()
plt.savefig(save_plots + 'refractive_index_combined.png')
