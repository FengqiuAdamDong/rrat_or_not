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
import smplotlib

class inject_stats_collection(inject_stats):
    def __init__(self):
        self.inj_stats = []
        self.folder = []
        self.det_snr = []
        self.detected_pulses = []
        self.snr = []
        self.det_frac = []
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
                print(f"{f}")
                inst.compare([csv], title=f)
                self.det_snr.append(inst.det_snr)
                self.detected_pulses.append(inst.detected_pulses)
                self.snr.append(inst.snr)
                self.det_frac.append(inst.det_frac)
        self.snr = np.array(self.snr)
        self.det_frac = np.array(self.det_frac)
        #fit this to polynomial
        #average along the 0th axis
        self.snr = np.mean(self.snr,axis=0)
        self.det_frac = np.mean(self.det_frac,axis=0)
        self.det_snr = np.array(self.det_snr)
        self.detected_pulses = np.array(self.detected_pulses)

        self.total_injections_per_snr = self.det_snr.shape[0]*self.det_snr.shape[2]
        self.inj_snr_fit = self.fit_poly(x=self.snr,p=self.det_frac,deg=7)
        all_det_snr = self.det_snr.flatten()
        detected_snr = self.det_snr[self.detected_pulses]
        self.bin_detections(all_det_snr, detected_snr, num_bins=30)
        self.poly_det_fit = self.fit_poly(x=self.detected_bin_midpoints,p=self.detected_det_frac,deg=7)
        predict_x_array = np.linspace(0,10,10000)
        self.predict_poly(predict_x_array,x=self.detected_bin_midpoints,p=self.detected_det_frac,plot=True,title="overall detection curve")
        detect_errors = np.array(list(inj_stats.detect_error_snr for inj_stats in self.inj_stats))
        self.detect_error_snr = np.sqrt(np.mean(detect_errors**2))
        print(self.detect_error_snr)
        self.deconvolve_response(self.det_frac, self.detect_error_snr, self.snr, self.inj_snr_fit)

        error_det_frac_d = np.sqrt(self.detected_det_frac*(1-self.detected_det_frac)/self.num_points_per_bin)
        # plt.scatter(self.detected_bin_midpoints,self.detected_det_frac,label=r"Measured $P(det|S_{det)$")
        plt.errorbar(self.detected_bin_midpoints,self.detected_det_frac,yerr=error_det_frac_d,fmt=".",label=r"Measured $P(det|S_{det})$")
        plt.legend()
        plt.xlim(0,5.5)
        plt.savefig("det_frac_fit.pdf")
        # plt.show()
        # plt.savefig("overall_detection_curve.png")
        # plt.close()

        # interp_p = np.interp(predict_x_array, self.detected_bin_midpoints, self.detected_det_frac)
        # plt.plot(predict_x_array, interp_p, label="interpolated")
        # plt.title("overall detection curve interp")
        # plt.legend()
        # plt.savefig("overall_detection_curve_interp.png")
        # plt.close()

    def model(self,X, snr_fit, det_error, snr, det_frac, deg=5,plot=False):
        #create a model for the detection curve
        predict_y = self.predict_poly(snr_fit,x=snr,p=det_frac,poly=X)
        #convolve with a gaussian
        gaussian = norm.pdf(snr_fit,loc=0,scale=det_error)
        convolved = convolve(predict_y,gaussian,mode="same")
        #scale to the max of det_frac
        convolved = convolved / np.max(convolved) * np.max(det_frac)
        #then do an interp for snr
        interp_p = np.interp(snr, snr_fit, convolved)
        if plot:
            error_det_frac_i = np.sqrt(det_frac*(1-det_frac)/self.total_injections_per_snr)
            plt.figure(figsize=(8,5))
            plt.errorbar(snr,det_frac,yerr=error_det_frac_i,fmt=".",label=r"$P(det|S_T)$")
            # plt.scatter(snr,det_frac,label=r"$P(det|SNR_t)$")
            plt.plot(snr_fit,predict_y,label=r"Forward model $P(det|S_{det})$")

            plt.xlabel("S/N")
            plt.ylabel("Probability")
            plt.axvline(2,linestyle="--",color="k")
            plt.xlim(0,5.5)
            plt.legend()
            plt.plot(snr,interp_p,label=r"Fitted $P(det|S_T)$")
            plt.savefig("det_frac.pdf")
        return interp_p

    def loglikelihood(self, X, snr_fit, det_error, snr, det_frac, deg=10):
        interp_p = self.model(X, snr_fit, det_error, snr, det_frac, deg=deg)
        #calculate the squared_difference
        squared_difference = np.log(np.exp(-(det_frac - interp_p)**2/2))
        return -np.sum(squared_difference)


    def deconvolve_response(self,det_frac, det_error, snr, poly):

        snr_fit = np.linspace(-10,10,1000)
        spacing = snr_fit[1] - snr_fit[0]
        #generate an array with the same spacing as the snr
        p_det_predict = self.predict_poly(snr_fit,x=snr,p=det_frac,poly=poly,plot=True,title="overall detection curve")
        #minimize the loglikelihood
        from scipy.optimize import minimize
        res = minimize(self.loglikelihood, x0=poly, args=(snr_fit, det_error, snr, det_frac, len(poly)), method='nelder-mead', options={'xatol': 1e-8,'maxiter':10000000, 'disp': True})
        print(res.x)
        predict_y = self.predict_poly(snr_fit,x=snr,p=det_frac,poly=res.x)
        self.model(res.x,snr_fit,det_error,snr,det_frac,plot=True)

        plt.figure(figsize=(8,5))
        plt.plot(snr_fit,predict_y,label="Forward model $P(det|S_{det})$")
        #add a line at x=2
        plt.axvline(x=2,linestyle="--",color="black")
        # plt.scatter(snr,det_frac,label=r"$P(det|S_t)$")
        plt.xlabel("S/N")
        plt.ylabel("Probability")
        self.detected_snr_fit = snr_fit
        self.detected_det_frac_fit = predict_y





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
