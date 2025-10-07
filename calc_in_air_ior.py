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

import sys
sys.path.append('/Users/nathanielalden/Downloads/RNO-G/t_abs_pulsing')
import delay_utils
import all_locs_ior

def unfold_tx_rx_antennas(tvals, sigvals, antenna_response):
    print(f"Need to get antenna response for this")

def find_time_from_template(tvals, sigvals, ch1, template_file = "raw_data/pulse_template.npy"):
    tvals, template = np.load(template_file)
    time_from_template_peak = delay_utils.find_delta_t(tvals, sigvals, template)

    time_from_zero = delay_utils.find_delta_t(tvals, ch1, template)

    #print(f"time delay from template matching is {time_from_template_peak-time_from_zero}")
    delta_t = time_from_template_peak-time_from_zero
    return delta_t

def find_ior(filename, rx_cable = None, tx_cable = None):
    script_dir = os.path.dirname(__file__)
    save_plots = os.path.join(script_dir, f"plots/")
    #save_plots = os.path.join(script_dir, f"{rootdir}/prel_analysis/results/{args.station}/{args.meas}/plots/")


    plt.figure(figsize=(10, 6))

    df = pd.read_csv(filename)
    column_map = {
        'time': 'in s',
        'ch1': 'C1 in V',
        'ch3': 'C3 in V'
    }

    time = df[column_map['time']].values
    ch1 = df[column_map['ch1']].values
    ch3 = df[column_map['ch3']].values
    
    print(time[np.argmax(ch3)], np.argmax(ch3), np.max(ch3), np.min(ch3))

    delta_t = all_locs_ior.find_time_delay(time, ch1, ch3)
    #print(f"Regular delta_T is {delta_t}")

    tmax = time[np.argmax(ch3)]
    plt.plot(time, ch3)
    template = np.copy(ch3)
    template[np.abs(time-tmax) > 1e-7] = 0

    new_delta_t = find_time_from_template(time, ch3, ch1)

    plt.plot(time[np.abs(time-tmax) < 1e-7], ch3[np.abs(time-tmax) < 1e-7])
    
    #np.save('raw_data/pulse_template.npy', [time, template])
    
    plt.savefig(save_plots + 'waveforms.png')    
    
    if (rx_cable is not None) and (tx_cable is not None):
        cable_a = delay_utils.get_cable_delay(tx_cable, cable = "TX")*1e-9
        cable_b = delay_utils.get_cable_delay(rx_cable, cable = "RX")*1e-9
        jumper_cable = 2.968e-9

        point_a = (72.628818467, -38.403947304)
        point_b = (72.629055327, -38.404391921)
        baseline = geodesic(point_a, point_b).meters
        ior = (c * (delta_t - cable_a - cable_b + jumper_cable)) / baseline
        print(f"The IOR is {ior}")

        ior = (c * (new_delta_t - cable_a - cable_b + jumper_cable)) / baseline
        print(f"The IOR from fancy template matching is {ior}")

    #print(f"depth is {depth}, ior is {ior}")



if __name__ == "__main__":
    #find_ior("raw_data/station_34/pulsing_b_c_down/SRF_T32.CSV")
    rx_cable = "raw_data/station_35/cableTDR_pulsing_in_air/SRF_T47.CSV"
    tx_cable = "raw_data/station_35/cableTDR_pulsing_in_air/SRF_T46.CSV"
    find_ior("raw_data/station_35/pulsing_in_air/SRF_T43.CSV", rx_cable = rx_cable, tx_cable=tx_cable)
    find_ior("raw_data/station_35/pulsing_in_air/SRF_T44.CSV", rx_cable = rx_cable, tx_cable=tx_cable)
    find_ior("raw_data/station_35/pulsing_in_air/SRF_T45.CSV", rx_cable = rx_cable, tx_cable=tx_cable)