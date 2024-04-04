#!/usr/bin/env python3

import numpy as np
import yaml
import argparse
import csv
from sigpyproc import readers as r
import os
import glob
import dill
import matplotlib.pyplot as plt
# creates the yaml file for pulsar
def process_detection_results(real_det):
    with open(real_det, "rb") as inf:
        det_class = dill.load(inf)

    det_fluence = []
    det_width = []
    det_snr = []
    noise_std = []
    print(f"total pulses: {len(det_class.sorted_pulses)}")
    not_fitted = 0
    for pulse_obj in det_class.sorted_pulses:
        if pulse_obj.det_amp != -1:
            det_fluence.append(pulse_obj.det_fluence)
            det_width.append(pulse_obj.det_std)
            det_snr.append(pulse_obj.det_snr)
            noise_std.append(pulse_obj.noise_std)
        else:
            not_fitted += 1
    print(f"not fitted: {not_fitted}")
    det_fluence = np.array(det_fluence)
    det_width = np.array(det_width)
    det_snr = np.array(det_snr)
    noise_std = np.array(noise_std)

    return det_fluence, det_width, det_snr, noise_std


parser = argparse.ArgumentParser(description="Create yaml file for pulsar")
parser.add_argument("csv_file", type=str, help="csv file with pulsar data")

args = parser.parse_args()
csv_file = args.csv_file

# read the csv file
pulsar_name = []
pulsar_period = []
with open(csv_file, "r") as f:
    reader = csv.reader(f)
    for row in reader:
        pulsar_name.append(row[0])
        pulsar_period.append(float(row[2]))

for pulsar in pulsar_name:
    # create the yaml file
    yaml_file = f"{pulsar}/fdp/{pulsar}.yaml"
    #check if yaml file already exists
    if os.path.exists(yaml_file):
        #load the yaml file
        with open(yaml_file, "r") as f:
            yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
            snr_thresh = yaml_dict["snr_thresh"]
            width_thresh = yaml_dict["width_thresh"]
            try:
                snr_upper = yaml_dict["snr_upper"]
            except KeyError:
                snr_upper = 50
            try:
                width_upper = yaml_dict["width_upper"]
            except KeyError:
                width_upper = float(28e-3)
    dill_file = f"{pulsar}/fdp/{pulsar}.dill"

    try:
        det_fluence, det_width, det_snr, noise_std = process_detection_results(dill_file)
    except:
        print(f"Error in processing {pulsar}")
        continue
    mask = (det_snr > snr_thresh) & (det_width > width_thresh) & (det_snr < snr_upper) & (det_width < width_upper)
    det_snr_masked = det_snr[mask]
    det_fluence_masked = det_fluence[mask]
    det_width_masked = det_width[mask]
    standard_mask = (det_snr > 1) & (det_width > 1e-3) & (det_snr < 50) & (det_width < 28e-3)
    det_snr_standard = det_snr[standard_mask]
    det_fluence_standard = det_fluence[standard_mask]
    det_width_standard = det_width[standard_mask]
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    try:
        bins = "auto"
        ax[0].hist(det_fluence_standard, bins=bins)
        ax[0].set_title(f"Detected Fluence, total: {len(det_fluence_standard)}")
        ax[0].set_xlabel("Fluence")
        ax[0].set_ylabel("Counts")
        ax[1].hist(det_width_standard, bins=bins)
        ax[1].set_title(f"Detected Width, total: {len(det_width_standard)}")
        ax[1].set_xlabel("Width")
        ax[1].set_ylabel("Counts")
        ax[2].hist(det_snr_standard, bins=bins)
        ax[2].set_title(f"Detected SNR, total: {len(det_snr_standard)}")
        ax[2].set_xlabel("SNR")
        ax[2].set_ylabel("Counts")
        plt.tight_layout()
        plt.savefig(f"{pulsar}/fdp/{pulsar}_detected.png")
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].hist(det_fluence_masked, bins=bins)
        ax[0].set_title(f"Detected Fluence, SNR > {snr_thresh}, Width > {width_thresh}, total: {len(det_fluence_masked)}")
        ax[0].set_xlabel("Fluence")
        ax[0].set_ylabel("Counts")
        ax[1].hist(det_width_masked, bins=bins)
        ax[1].set_title(f"Detected Width, SNR > {snr_thresh}, Width > {width_thresh}, total: {len(det_width_masked)}")
        ax[1].set_xlabel("Width")
        ax[1].set_ylabel("Counts")
        ax[2].hist(det_snr_masked, bins=bins)
        ax[2].set_title(f"Detected SNR, SNR > {snr_thresh}, Width > {width_thresh}, total: {len(det_snr_masked)}")
        ax[2].set_xlabel("SNR")
        ax[2].set_ylabel("Counts")
        plt.tight_layout()
        plt.savefig(f"{pulsar}/fdp/{pulsar}_detected_masked.png")
        print(f"plot saved to {pulsar}/fdp/{pulsar}_detected.png")
        plt.show()
    except:
        print(f"plots not made for {pulsar}")
        continue
