import dill
import matplotlib.pyplot as plt
import cupy as cp
import numpy as np
import scipy
import pandas as pd
class statistics_basic:
    def __init__(self,detection_curve,flux_cal=1,plot=True):
        self.load_detection_fn(detection_curve,flux_cal)

    def p_detect(self,snr,min_snr_cutoff=1.6,flux_cal=1):
        interp_res = np.interp(snr,inj_stats.detected_bin_midpoints,inj_stats.detected_det_frac)
        # interp_res = np.interp(snr,inj_stats.detected_snr_fit,inj_stats.detected_det_frac_fit)

        #remmove those values below snr=1.3
        snr_cutoff = snr[np.where(interp_res>0)[0][0]]
        if snr_cutoff < min_snr_cutoff:
            snr_cutoff = min_snr_cutoff
        inj_stats._snr = snr*flux_cal
        inj_stats._interp_res = interp_res
        #scale the cutoff by the flux cal
        snr_cutoff = snr_cutoff*flux_cal
        inj_stats._interp_res[inj_stats._snr<snr_cutoff] = 0

        return interp_res

    def p_detect_cpu(self,points,fluence=False):
        if fluence:
            interp_res = self.detected_interp_fluence(points)
        else:
            interp_res = self.detected_interp_snr(points)
        interp_res[interp_res<0] = 0
        interp_res[interp_res>1] = 1
        return interp_res


    def p_detect_cupy(self,snr,interpolator):
        interp_res = cp.interp(snr,cp.array(inj_stats._snr),cp.array(inj_stats._interp_res))
        # interp_res = cp.interp(snr,cp.array(inj_stats.detected_bin_midpoints),cp.array(inj_stats.detected_det_frac))

        # interp_res = cp.interp(snr,cp.array(inj_stats.detected_snr_fit),cp.array(inj_stats.detected_det_frac_fit))

        #remmove those values below snr=1.3
        # interp_res[snr<1.3] = 0
        return interp_res

    def load_detection_fn(self,detection_curve,plot=True,min_snr_cutoff=1.6,flux_cal=1):
        with open(detection_curve, "rb") as inf:
            inj_stats = dill.load(inf)

        detected_snr_bins = inj_stats.detected_bin_midpoints_snr[0]
        detected_width_bins = inj_stats.detected_bin_midpoints_snr[1]
        detected_det_frac_snr = inj_stats.detected_det_frac_snr
        self.detected_interp_snr = scipy.interpolate.RegularGridInterpolator((detected_snr_bins,detected_width_bins),detected_det_frac_snr,bounds_error=False,fill_value=None)

        detected_fluence_bins = inj_stats.detected_bin_midpoints_fluence[0]
        detected_width_f_bins = inj_stats.detected_bin_midpoints_fluence[1]
        detected_det_frac_fluence = inj_stats.detected_det_frac_fluence
        detected_det_frac_fluence = pd.DataFrame(detected_det_frac_fluence)
        detected_det_frac_fluence.interpolate(limit_direction='both',axis=0,inplace=True)
        #convert back to numpy array
        detected_det_frac_fluence = detected_det_frac_fluence.to_numpy()
        self.detected_interp_fluence = scipy.interpolate.RegularGridInterpolator((detected_fluence_bins,detected_width_f_bins),detected_det_frac_fluence,bounds_error=False,fill_value=None)

        injected_snr = inj_stats.unique_snrs
        injected_width = inj_stats.unique_widths
        injected_det_frac = inj_stats.det_frac_matrix_snr
        self.injected_interp_snr = scipy.interpolate.RegularGridInterpolator((injected_snr,injected_width),injected_det_frac,bounds_error=False,fill_value=None)


        snr_arr = np.linspace(0, 6, 1000)
        width_arr = np.linspace(1, 20, 1000)*1e-3
        fluence_arr = np.linspace(0, 1, 1000)
        snr_grid, width_grid = np.meshgrid(snr_arr, width_arr, indexing="ij")
        points = (snr_grid, width_grid)
        interp_res_snr = self.p_detect_cpu(points,fluence=False)
        fluence_grid, width_grid = np.meshgrid(fluence_arr, width_arr, indexing="ij")
        points = (fluence_grid, width_grid)
        interp_res_fluence = self.p_detect_cpu(points,fluence=True)

        # detfn = p_detect(snr_arr,min_snr_cutoff=min_snr_cutoff,flux_cal=flux_cal)
        #get the snr cutoff by finding when detfn is larger than 0.05
        # snr_cutoff = inj_stats._snr[np.where(detfn>0)[0][0]]
        if plot:
            fig, ax = plt.subplots(1,2,figsize=(10,5))
            mesh = ax[0].pcolormesh(width_grid*1e3, snr_arr, interp_res_snr)
            ax[0].set_xlabel("width")
            ax[0].set_ylabel("snr")
            cbar = plt.colorbar(mesh,ax=ax[0])
            cbar.set_label("detection fraction")
            mesh = ax[1].pcolormesh(detected_width_bins*1e3, detected_snr_bins,detected_det_frac_snr)
            ax[1].set_xlabel("width")
            ax[1].set_ylabel("snr")
            cbar = plt.colorbar(mesh,ax=ax[1])
            cbar.set_label("detection fraction")

            fig, ax = plt.subplots(1,2,figsize=(10,5))
            mesh = ax[0].pcolormesh(width_grid*1e3, fluence_arr, interp_res_fluence)
            ax[0].set_xlabel("width")
            ax[0].set_ylabel("fluence")
            cbar = plt.colorbar(mesh,ax=ax[0])
            cbar.set_label("detection fraction")
            mesh = ax[1].pcolormesh(detected_width_f_bins*1e3, detected_fluence_bins, inj_stats.detected_det_frac_fluence)
            ax[1].set_xlabel("width")
            ax[1].set_ylabel("fluence")
            cbar = plt.colorbar(mesh,ax=ax[1])
            cbar.set_label("detection fraction")
            plt.show()

        snr_cutoff = min_snr_cutoff
        # if snr_cutoff < 1.3:
        #     print("WARNING SNR CUTOFF IS LESS THAN 1.3")
        #     snr_cutoff = 1.3
        #assign the errors in the different directions
        self.detected_error_snr = inj_stats.detect_error_snr
        self.detected_error_width = inj_stats.detect_error_width
        self.detected_error_fluence = inj_stats.detect_error_fluence
        return snr_cutoff

    def logistic(self,x, k, x0):
        L = 1
        snr = x
        detection_fn = np.zeros(len(snr))
        snr_limit = 1
        detection_fn[(snr > -snr_limit) & (snr < snr_limit)] = L / (
            1 + np.exp(-k * (snr[(snr > -snr_limit) & (snr < snr_limit)] - x0))
        )
        detection_fn[snr >= snr_limit] = 1
        detection_fn[snr <= -snr_limit] = 0
        return detection_fn


    def n_detect(self,snr_emit):
        # snr emit is the snr that the emitted pulse has
        p = p_detect(snr_emit)
        # simulate random numbers between 0 and 1
        rands = np.random.rand(len(p))
        # probability the random number is less than p gives you an idea of what will be detected
        detected = snr_emit[rands < p]
        return detected
