import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

def resample(tvals, sig, target_dt):
    
    source_dt = tvals[3] - tvals[2]
    assert target_dt < source_dt

    target_tvals = np.linspace(tvals[0], tvals[-1], int((tvals[-1] - tvals[0]) / target_dt))
    os_factor = int(source_dt / target_dt) + 4

    # oversample the original waveform
    os_length = os_factor * len(sig)
    os_sig = signal.resample(sig, os_length)
    os_tvals = np.linspace(tvals[0], tvals[-1] + tvals[1] - tvals[0], os_length, endpoint = False)

    # evaluate the oversampled waveform on the target grid
    target_sig = np.interp(target_tvals, os_tvals, os_sig)
    return target_tvals, target_sig

def find_delta_t(tvals, sigvals, sigvals_ref):
    fs = 1.0 / (tvals[1] - tvals[0])    

    corr = np.correlate(sigvals, sigvals_ref, mode = "full")
    tvals_corr = np.linspace(-len(corr) / (2 * fs), len(corr) / (2 * fs), len(corr))

    return tvals_corr[np.argmax(corr)]

def get_cable_delay(filename, cable = None):
    df = pd.read_csv(filename, names=['t', 'C1', 'C3'], skiprows = 1)

    df['t'] *= 1e9
    df['template_V'] = df['C1']
    df['reflection_V'] = df['C1']

    df.loc[np.abs(df['t']) > 2, 'template_V'] = 0
    df.loc[np.abs(df['t']) < 2, 'reflection_V'] = 0

    dt = df['t'][1]-df['t'][0]
    up_factor = 10
    t_up, template_up = resample(df['t'].to_numpy(), df['template_V'].to_numpy(), dt/up_factor)
    _, reflection_up = resample(df['t'].to_numpy(), df['reflection_V'].to_numpy(), dt/up_factor)

    upsample_delta_t = find_delta_t(t_up, reflection_up, template_up)
    delta_t = find_delta_t(df['t'], df['reflection_V'], df['template_V'])

    if cable is not None:
        print(f"upsampled delta_t is {upsample_delta_t/2} for {cable} cable")

    '''
    plt.plot(df['t'], df['template_V'], label = "template_V")
    plt.plot(df['t'] , df['reflection_V'], label = "reflection_V")

    plt.legend()
    plt.show()
    #plt.savefig("TDR_vis.pdf")
    '''

    return upsample_delta_t/2

if __name__ == "__main__":
    tx_file = "SRF_T46.CSV"
    rx_file = "SRF_T47.CSV"
    jumper_file = "SRF_T48.CSV"
    get_cable_delay(f"050825/cableTDR/{rx_file}", cable = "RX")
    get_cable_delay(f"050825/cableTDR/{tx_file}", cable = "TX")
    get_cable_delay(f"050825/cableTDR/{jumper_file}", cable = "Jumper")


