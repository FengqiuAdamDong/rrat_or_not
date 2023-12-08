#!/usr/bin/env python3
import numpy as np
import sys
import glob
import os

def check_injection_status(sample):
    grid = sample["grid"]
    snrs = grid[:, 1]
    widths = grid[:, 3]
    unique_snrs = np.array(list(set(snrs)))
    unique_snrs = np.array(list(str(np.around(s, 4)).zfill(6) for s in unique_snrs))
    unique_widths = np.array(list(set(widths)))
    unique_widths = np.array(list(str(np.around(w, 4)).zfill(6) for w in unique_widths))
    # check file existance
    fil_files = glob.glob("*SNR*.fil")
    # check all the unique snrs are in there
    all_injected = np.zeros((len(unique_snrs),len(unique_widths)), dtype=bool)
    for f in fil_files:
        for j, s in enumerate(unique_snrs):
            for k, w in enumerate(unique_widths):
                if f"SNR{s}_width{w}.fil" in f:
                    all_injected[j,k] = True
    #flatten all_injected
    all_injected = all_injected.flatten()
    if sum(all_injected) == len(all_injected):
        return True
    else:
        return False


if __name__ == "__main__":
    if os.path.exists(sys.argv[1]):
        sample = np.load(sys.argv[1], allow_pickle=1)
        injection_complete = check_injection_status(sample)
        if injection_complete:
            print("0")
        else:
            print("1")
    else:
        print("1")
