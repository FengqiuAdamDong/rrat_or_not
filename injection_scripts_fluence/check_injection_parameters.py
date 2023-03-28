#!/usr/bin/env python3

# THIS SCRIPT AIMS TO USE inj_stats.dill TO PLOT THE ACTUAL INJECTED SNR VS DETECTED SNR OF THE INJECTIONS
import dill
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    with open("inj_stats.dill", "rb") as inj:
        inj_stats = dill.load(inj)
    sorted_injects = inj_stats.sorted_inject
    detected_snrs = list(np.mean(d.det_snr) for d in sorted_injects)
    detected_snr_sigma = list(np.std(d.det_snr) for d in sorted_injects)
    snrs = list(d.snr for d in sorted_injects)
    plt.errorbar(snrs, detected_snrs, yerr=detected_snr_sigma, fmt="o")
    plt.show()
    import pdb

    pdb.set_trace()
