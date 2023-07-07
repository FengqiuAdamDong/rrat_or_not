#!/usr/bin/env python3

###THIS SCRIPT WILL go through every file and combine the inj_stats to give you an overall inj_stats
import numpy as np
import matplotlib.pyplot as plt
import sys
import dill
from inject_stats import inject_stats


class inject_stats_collection(inject_stats):
    def __init__(self):
        self.inj_stats = []
        self.folder = []
        self.det_snr = []
        self.detected_pulses = []
    def calculate_detection_curve(self, csvs="1"):
        # build statistics
        snrs = []
        detecteds = []
        totals = []
        for inst, f in zip(self.inj_stats, self.folder):
            if csvs != "all":
                # only compare csv_1
                csv = f"{f}/positive_bursts_1.csv"
                inst.get_base_fn()
                inst.amplitude_statistics()
                inst.compare([csv], title=f)
                self.det_snr.append(inst.det_snr)
                self.detected_pulses.append(inst.detected_pulses)
        self.det_snr = np.array(self.det_snr)
        self.detected_pulses = np.array(self.detected_pulses)
        all_det_snr = self.det_snr.flatten()
        detected_snr = self.det_snr[self.detected_pulses]
        self.bin_detections(all_det_snr, detected_snr, num_bins=30)
        self.poly_det_fit = self.fit_poly(x=self.detected_bin_midpoints,p=self.detected_det_frac,deg=50)
        predict_x_array = np.linspace(0,10,10000)
        self.predict_poly(predict_x_array,x=self.detected_bin_midpoints,p=self.detected_det_frac,plot=True,title="overall detection curve")
        detect_errors = list(inj_stats.detect_error_snr for inj_stats in self.inj_stats)
        self.detect_error_snr = np.mean(detect_errors)
        plt.savefig("overall_detection_curve.png")
        plt.close()
        interp_p = np.interp(predict_x_array, self.detected_bin_midpoints, self.detected_det_frac)
        plt.plot(predict_x_array, interp_p, label="interpolated")
        plt.title("overall detection curve interp")
        plt.legend()
        plt.savefig("overall_detection_curve_interp.png")
        plt.close()


def combine_images():
    import os
    import glob
    from PIL import Image

    image_array = glob.glob("*fit_snr.png")
    images = [Image.open(x) for x in image_array]
    widths, heights = zip(*(i.size for i in images))
    row_len = 10
    total_width = widths[0] * row_len
    max_height = (int(len(heights) / row_len) + 1) * heights[0]

    new_im = Image.new("RGB", (total_width, max_height))

    x_offset = 0
    y_offset = 0
    for i, im in enumerate(images):
        new_im.paste(im, (x_offset, y_offset))
        if (i > 0) & ((i % row_len) == 0):
            y_offset += im.size[1]
            x_offset = 0
        else:
            x_offset += im.size[0]

    new_im.save("detection_curves_all_combined.jpg")


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

    import dill
    with open("inj_stats_combine_fitted.dill", "wb") as of:
        dill.dump(inj_collection, of)

    combine_images()
