import numpy as np
import sys
import matplotlib.pyplot as plt
import dill

fitted_inj_list = sys.argv[1:]

for i,inj_file in enumerate(fitted_inj_list):
    with open(inj_file, 'rb') as f:
        inj_stats = dill.load(f)
    unique_snrs = inj_stats.unique_snrs
    unique_widths = inj_stats.unique_widths
    det_frac_snr = inj_stats.det_frac_matrix_snr
    if i == 0:
        columns = 5
        rows = int(len(unique_widths)/5)
        fig, ax = plt.subplots(rows, columns, figsize=(20,25))
    for j, width in enumerate(unique_widths):
        row = j//columns
        col = j%columns
        ax[row, col].plot(unique_snrs, det_frac_snr[:,j], label=inj_file)
        ax[row, col].set_title('Width: %s'%width)
        ax[row, col].set_xlabel('SNR')
        ax[row, col].set_ylabel('Detection Fraction')
        ax[row, col].legend()
plt.tight_layout()

#plotting the detected det frac
for i,inj_file in enumerate(fitted_inj_list):
    with open(inj_file, 'rb') as f:
        inj_stats = dill.load(f)
    unique_snrs = inj_stats.detected_bin_midpoints_snr[0]
    unique_widths = inj_stats.detected_bin_midpoints_snr[1]
    det_frac_snr = inj_stats.detected_det_frac_snr
    if i == 0:
        columns = 5
        rows = int(len(unique_widths)/5)
        fig, ax = plt.subplots(rows, columns, figsize=(20,25))
    for j, width in enumerate(unique_widths):
        row = j//columns
        col = j%columns
        ax[row, col].plot(unique_snrs, det_frac_snr[:,j], label=inj_file)
        ax[row, col].set_title('Width: %s'%width)
        ax[row, col].set_xlabel('SNR')
        ax[row, col].set_ylabel('Detection Fraction')
        ax[row, col].legend()
plt.tight_layout()
plt.show()
