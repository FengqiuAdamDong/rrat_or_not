#!/usr/bin/env python3
import numpy as np
import sys
import glob
import os


def check_injection_status(sample):
    grid = sample["grid"]
    snrs = grid[:, 1]
    unique_snrs = np.array(list(set(snrs)))
    unique_snrs = np.array(list(np.round(s, 2) for s in unique_snrs))
    # check file existance
    fil_files = glob.glob("*snr*.fil")
    # check all the unique snrs are in there
    all_injected = np.zeros(len(unique_snrs), dtype=bool)
    for f in fil_files:
        for j, s in enumerate(unique_snrs):
            if f"snr{s}.fil" in f:
                all_injected[j] = True
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
