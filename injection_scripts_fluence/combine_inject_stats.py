#!/usr/bin/env python3

###THIS SCRIPT WILL go through every file and combine the inj_stats to give you an overall inj_stats
import numpy as np
import matplotlib.pyplot as plt
import sys
import dill
from inject_stats import inject_stats
from scipy.signal import deconvolve
from scipy.signal import convolve
from scipy.stats import norm
import scipy.fft as fft
# import smplotlib
from inject_stats import create_matrix


class inject_stats_collection(inject_stats):
    def __init__(self):
        self.inj_stats = []
        self.folder = []
        self.det_snr = []
        self.detected_pulses = []
        self.inj_snr = []
        self.det_frac_matrix_snr = []
        self.det_width = []
        self.inj_width = []
        self.det_fluence = []
        self.detect_error_snr_arr = []
        self.detect_error_width_arr = []
        self.detect_error_fluence_arr = []
        self.detect_error_snr_low_width_arr = []
        self.detect_error_width_low_width_arr = []
    def calculate_detection_curve(self, csvs="1"):
        # build statistics
        snrs = []
        detecteds = []
        totals = []
        for inst, f in zip(self.inj_stats, self.folder):
            if csvs != "all":
                # only compare csv_1
                csv = f"{f}/positive_bursts_1.csv"
                print(f)
                inst.set_base_fn(f)
                inst.amplitude_statistics(title=f)
                if inst.detect_error_snr > 0.5:
                    print(f"skipping {f} because of detect_error_snr")
                    continue
                if inst.detect_error_width > 2.5e-3:
                    print(f"skipping {f} because of detect_error_width")
                    continue

                inst.compare([csv], title=f+"_det_curve")
                self.detect_error_snr_arr.append(inst.detect_error_snr)
                self.detect_error_width_arr.append(inst.detect_error_width)
                self.detect_error_fluence_arr.append(inst.detect_error_fluence)
                self.detect_error_snr_low_width_arr.append(inst.detect_error_snr_low_width)
                self.detect_error_width_low_width_arr.append(inst.detect_error_width_low_width)

                plt.close("all")
                for si in inst.sorted_inject:
                    self.inj_snr.append(si.snr)
                    self.inj_width.append(si.width)

                    self.detected_pulses.append(si.detected)

                    self.det_snr.append(si.det_snr)
                    self.det_width.append(si.det_std)
                    if -1 in si.det_snr:
                        #Something obviously went wrong, lets figure out why
                        import pdb; pdb.set_trace()



                    self.det_fluence.append(si.det_fluence)
        self.detected_pulses = np.array(self.detected_pulses).flatten()

        self.det_snr = np.array(self.det_snr).flatten()
        self.inj_snr = np.array(self.inj_snr).flatten()

        self.det_width = np.array(self.det_width).flatten()
        self.inj_width = np.array(self.inj_width).flatten()

        self.det_fluence = np.array(self.det_fluence).flatten()
        self.inj_fluence = self.inj_snr * self.inj_width/0.3989

        self.detect_error_snr_arr = np.array(self.detect_error_snr_arr)
        self.detect_error_width_arr = np.array(self.detect_error_width_arr)
        self.detect_error_fluence_arr = np.array(self.detect_error_fluence_arr)
        self.detect_error_snr_low_width_arr = np.array(self.detect_error_snr_low_width_arr)
        self.detect_error_width_low_width_arr = np.array(self.detect_error_width_low_width_arr)
        #filter out the outliers of detect_error_snr_arr
        # print("filtering out # of outliers: ", np.sum(self.detect_error_snr_arr > 0.5))
        self.detect_error_snr = np.sqrt(np.mean((self.detect_error_snr_arr[self.detect_error_snr_arr < 0.5])**2))
        self.detect_error_width = np.sqrt(np.mean((self.detect_error_width_arr[self.detect_error_snr_arr < 0.5])**2))
        self.detect_error_fluence = np.sqrt(np.mean(self.detect_error_fluence_arr[self.detect_error_snr_arr < 0.5]**2))
        self.detect_error_snr_low_width = np.sqrt(np.mean(self.detect_error_snr_low_width_arr[self.detect_error_snr_arr < 1]**2))
        self.detect_error_width_low_width = np.sqrt(np.mean(self.detect_error_width_low_width_arr[self.detect_error_snr_arr < 1]**2))

        #create a matrix of the detection fraction
        unique_snr = np.unique(self.inj_snr)
        unique_width = np.unique(self.inj_width)
        self.det_frac_matrix_snr = np.zeros((len(unique_snr), len(unique_width)))
        print(self.det_frac_matrix_snr.shape)
        for i, snr in enumerate(unique_snr):
            for j, width in enumerate(unique_width):
                mask = (self.inj_snr == snr) & (self.inj_width == width)
                self.det_frac_matrix_snr[i, j] = np.sum(self.detected_pulses[mask]) / np.sum(mask)

        detected_det_vals = self.det_snr[self.detected_pulses]
        detected_width_vals = self.det_width[self.detected_pulses]
        detected_fluence_vals = self.det_fluence[self.detected_pulses]

        self.bin_detections_2d(self.det_snr, detected_det_vals, self.det_width, detected_width_vals, num_bins=40,fluence=False)
        self.bin_detections_2d(self.det_snr, detected_det_vals, self.det_fluence, detected_fluence_vals, num_bins=40,fluence=True)

        #define the same values as the inj_stats.compare function
        self.unique_snrs = unique_snr
        self.unique_widths = unique_width
        self.detected_widths = detected_width_vals
        self.detected_amplitudes_snr = detected_det_vals
        self.detected_amplitudes_fluence = detected_fluence_vals
        self.all_det_amplitudes_snr = self.det_snr
        self.all_det_widths = self.det_width
        self.all_det_amplitudes_fluence = self.det_fluence



        fig, axes = plt.subplots(1, 2, figsize=(10, 10))
        mesh = axes[0].pcolormesh(unique_width*1000, unique_snr, self.det_frac_matrix_snr, cmap="viridis")
        mesh.set_clim(0, 1)
        cbar = plt.colorbar(mesh)
        cbar.set_label("Detection Fraction")
        axes[0].set_xlabel("Injected Width (ms)")
        axes[0].set_ylabel("Injected SNR")
        axes[0].set_title("Detection Fraction inj")

        mesh = axes[1].pcolormesh(self.detected_bin_midpoints_snr[1]*1000,self.detected_bin_midpoints_snr[0],
                                  self.detected_det_frac_snr, cmap="viridis")
        mesh.set_clim(0, 1)
        cbar = plt.colorbar(mesh)
        cbar.set_label("Detection Fraction")
        axes[1].set_xlabel("Detected Width (ms)")
        axes[1].set_ylabel("Detected SNR")
        axes[1].set_title("Detection Fraction det")
        plt.tight_layout()
        plt.savefig("detection_curves_all.png")
        plt.show()






# All inputs are
if __name__ == "__main__":
    fil_files = sys.argv[1:]
    inj_collection = inject_stats_collection()
    for i, f in enumerate(fil_files):
        folder_name = f.replace(".fil", "")
        try:
            with open(folder_name + "/inj_stats.dill", "rb") as inf:
                inj_stats = dill.load(inf)
                inj_stats = inject_stats(**inj_stats.__dict__)
                inj_stats.repopulate_io()

                inj_collection.inj_stats.append(inj_stats)
                inj_collection.folder.append(folder_name)
        except Exception as e:
            # for whatever reason, this failed, lets write it out and just move on
            print(f"failed on {f} with error {e}")
            continue

    inj_collection.calculate_detection_curve()
    inj_collection.forward_model_det()
    inj_collection.generate_forward_model_grid()
    import dill
    with open("inj_stats_combine_fitted.dill", "wb") as of:
        dill.dump(inj_collection, of)

    # combine_images()
