import dill
import matplotlib.pyplot as plt
import cupy as cp
from cupyx.scipy import interpolate as cpinterp
import numpy as np
import scipy
import pandas as pd
from scipy.stats import norm


class statistics_basic:
    def __init__(
        self, detection_curve, flux_cal=1, snr_cutoff=2, width_cutoff=5e-3, plot=True
    ):
        self.load_detection_fn(
            detection_curve,
            flux_cal=flux_cal,
            snr_cutoff=snr_cutoff,
            width_cutoff=width_cutoff,
            plot=plot,
        )

    def p_detect_cpu(self, points, fluence=False):
        if fluence:
            interp_res = self.detected_interp_fluence(points)
        else:
            interp_res = self.detected_interp_snr(points)
        interp_res[interp_res < 0] = 0
        interp_res[interp_res > 1] = 1
        return interp_res

    def p_detect_cpu_injected(self, points, fluence=False):
        if fluence:
            interp_res = self.injected_interp_fluence(points)
        else:
            interp_res = self.injected_interp_snr(points)
        interp_res[interp_res < 0] = 0
        interp_res[interp_res > 1] = 1
        return interp_res

    def p_detect_cpu_true_cupy(self, points):
        interp_res = self.cupy_pdet_st_wt_interp(points)
        interp_res[interp_res < 0] = 0
        interp_res[interp_res > 1] = 1
        return interp_res

    def p_detect_cpu_true(self, points, fluence=False):
        interp_res = self.pdet_st_wt_interp(points)
        interp_res[interp_res < 0] = 0
        interp_res[interp_res > 1] = 1
        return interp_res

    def p_detect_cupy(self, points, fluence=False, plot=True):
        if fluence:
            interp_res = self.cupy_detected_interp_fluence(points)
        else:
            interp_res = self.cupy_detected_interp_snr(points)
        interp_res[interp_res < 0] = 0
        interp_res[interp_res > 1] = 1
        return interp_res

    def convolve_p_detect(self, plot=True):
        # convolve the p_detect with the injected distribution
        # gaussian with sigma_amp_error
        true_snr_bins = np.linspace(0, 20, 500)
        true_snr_bins = true_snr_bins[np.newaxis, np.newaxis, :]
        true_width_bins = np.linspace(0, 60, 501) * 1e-3

        detected_snr_bins = np.linspace(0, 25, 502)
        detected_width_bins = np.linspace(0, 65, 503) * 1e-3
        detected_snr_bins = detected_snr_bins[:, np.newaxis, np.newaxis]
        detected_width_bins = detected_width_bins[np.newaxis, :, np.newaxis]
        points_det = (detected_snr_bins, detected_width_bins)
        p_det_snr = self.p_detect_cpu(points_det)
        amp_error = norm.pdf(detected_snr_bins, true_snr_bins, self.detected_error_snr)
        # form the 3d array
        pdet_st_sd_wd = p_det_snr * amp_error
        # marginalize over the sd
        pdet_st_wd = np.trapz(pdet_st_sd_wd, detected_snr_bins, axis=0)
        # plt.figure()
        # plt.pcolormesh(true_snr_bins[0,0,:],detected_width_bins[0,:,0]*1e3,pdet_st_wd)
        # plt.figure()
        # plt.pcolormesh(detected_snr_bins[:,0,0],detected_width_bins[0,:,0]*1e3,p_det_snr[:,:,0].T)
        # plt.figure()
        # plt.plot(detected_snr_bins[:,0,0],p_det_snr[:,250,0],label='detected')
        # plt.plot(true_snr_bins[0,0,:],pdet_st_wd[250,:],label='injected')
        # plt.legend()
        # plt.show()
        detected_width_bins = detected_width_bins[0, :, :, np.newaxis]
        width_error = norm.pdf(
            detected_width_bins, true_width_bins, self.detected_error_width
        )
        # form the 3d array
        pdet_st_wt_wd = pdet_st_wd[:, :, np.newaxis] * width_error
        # marginalize over the wd
        pdet_st_wt = np.trapz(pdet_st_wt_wd, detected_width_bins, axis=0)
        # create and interpolator
        self.pdet_st_wt_interp = scipy.interpolate.RegularGridInterpolator(
            (true_snr_bins[0, 0, :], true_width_bins),
            pdet_st_wt,
            bounds_error=False,
            fill_value=None,
        )
        self.cupy_pdet_st_wt_interp = cpinterp.RegularGridInterpolator(
            (cp.array(true_snr_bins[0, 0, :]), cp.array(true_width_bins)),
            cp.array(pdet_st_wt),
            bounds_error=False,
            fill_value=None,
        )
        self.pdet_st_wt = pdet_st_wt
        self.true_snr_bins = true_snr_bins[0, 0, :]
        self.true_width_bins = true_width_bins

        if plot:
            # plot this against the injected distribution
            fig, ax = plt.subplots(1, 3, figsize=(10, 5))
            mesh = ax[0].pcolormesh(
                true_snr_bins[0, 0, :], true_width_bins * 1e3, pdet_st_wt.T
            )
            mesh.set_clim(0, 1)
            ax[0].set_xlabel("Injected SNR")
            ax[0].set_ylabel("Injected Width (ms)")
            ax[0].set_title("interpolated")
            true_snr_bins = true_snr_bins[0, 0, :, np.newaxis]
            true_width_bins = true_width_bins[np.newaxis, :]
            points = (true_snr_bins, true_width_bins)
            interp_inj_mesh = self.p_detect_cpu_injected(points)
            mesh = ax[1].pcolormesh(
                true_snr_bins[:, 0], true_width_bins[0, :] * 1e3, interp_inj_mesh.T
            )
            mesh.set_clim(0, 1)
            ax[1].set_xlabel("Injected SNR")
            ax[1].set_ylabel("Injected Width (ms)")
            ax[1].set_title("injected value")
            test_snr = np.linspace(0, 20, 100)
            test_width = np.linspace(0, 60, 100) * 1e-3
            points = (test_snr[:, np.newaxis], test_width[np.newaxis, :])
            test_pdet_st_wt = self.p_detect_cpu_true(points)
            mesh = ax[2].pcolormesh(test_snr, test_width * 1e3, test_pdet_st_wt.T)
            mesh.set_clim(0, 1)
            ax[2].set_xlabel("Injected SNR")
            ax[2].set_ylabel("Injected Width (ms)")
            ax[2].set_title("interpolated injected value")
            # set ax2 xlim and ylim to be the same as ax0
            ax[2].set_xlim(ax[0].get_xlim())
            ax[2].set_ylim(ax[0].get_ylim())
            plt.tight_layout()

            fig, ax = plt.subplots(1, 3, figsize=(10, 5))
            ax[0].plot(true_snr_bins[:, 0], interp_inj_mesh[:, 400], label="injected")
            ax[0].plot(true_snr_bins[:, 0], pdet_st_wt[:, 400], label="convolved")
            ax[0].set_xlim(min(self.injected_snr), max(self.injected_snr))
            ax[0].legend()
            ax[0].set_xlabel("Injected SNR")

            ax[1].plot(
                true_width_bins[0, :] * 1e3, interp_inj_mesh[400, :], label="injected"
            )
            ax[1].plot(true_width_bins[0, :] * 1e3, pdet_st_wt[400, :], label="convolved")
            ax[1].set_xlim(min(self.injected_width) * 1e3, max(self.injected_width) * 1e3)

            ax[1].legend()
            ax[1].set_xlabel("Injected Width (ms)")
            ax[2].plot(detected_width_bins[:, 0, 0] * 1e3,p_det_snr[250, : , 0], label="detected")
            ax[2].set_xlabel("detected width (ms)")


            plt.show()

    def load_detection_fn(
        self, detection_curve, plot=True, snr_cutoff=2.0, width_cutoff=5e-3, flux_cal=1, use_interp=True
    ):
        with open(detection_curve, "rb") as inf:
            inj_stats = dill.load(inf)

        if use_interp:
            detected_snr_bins = inj_stats.forward_model_snr_arrs
            detected_width_bins = inj_stats.unique_widths
            detected_det_frac_snr = inj_stats.det_frac_foreward_model_matrix_snr
            # detected_snr_bins = inj_stats.unique_snrs
            # detected_width_bins = inj_stats.unique_widths
            # detected_det_frac_snr = inj_stats.det_frac_matrix_snr
        else:
            detected_snr_bins = inj_stats.detected_bin_midpoints_snr[0]
            detected_width_bins = inj_stats.detected_bin_midpoints_snr[1]
            detected_det_frac_snr = inj_stats.detected_det_frac_snr

        self.detected_interp_snr = scipy.interpolate.RegularGridInterpolator(
            (detected_snr_bins, detected_width_bins),
            detected_det_frac_snr,
            bounds_error=False,
            fill_value=None,
        )
        # do a stage of this interpolation process so that the interpolated cut-off is at the right place
        # this is only needed if the injected grid is not really fine
        detected_snr_bins_stage1 = np.linspace(0, max(detected_snr_bins), 1000)
        detected_width_bins_stage1 = np.linspace(0,35e-3 , 1000)
        detected_det_frac_snr_stage1 = self.p_detect_cpu(
            (
                detected_snr_bins_stage1[:, np.newaxis],
                detected_width_bins_stage1[np.newaxis, :],
            )
        )
        detected_det_frac_snr_stage1[
            np.argwhere(detected_snr_bins_stage1 < snr_cutoff), :
        ] = 0
        detected_det_frac_snr_stage1[
            :, np.argwhere(detected_width_bins_stage1 < width_cutoff)
        ] = 0
        detected_det_frac_snr_stage1[
            :, np.argwhere(detected_width_bins_stage1 >30e-3)
        ] = 0
        #also remove the corner of width<5e-3 and snr<2.8
        del_mask = (detected_snr_bins_stage1[:, np.newaxis] < 2.8) & (detected_width_bins_stage1[np.newaxis, :] < 5e-3)
        detected_det_frac_snr_stage1[del_mask] = 0
        self.detected_interp_snr = scipy.interpolate.RegularGridInterpolator(
            (detected_snr_bins_stage1, detected_width_bins_stage1),
            detected_det_frac_snr_stage1,
            bounds_error=False,
            fill_value=None,
        )


        detected_det_frac_snr[np.argwhere(detected_snr_bins < snr_cutoff), :] = 0
        detected_det_frac_snr[:, np.argwhere(detected_width_bins < width_cutoff)] = 0
        detected_det_frac_snr[:, np.argwhere(detected_width_bins >30e-3 )] = 0


        detected_fluence_bins = inj_stats.detected_bin_midpoints_fluence[0]
        detected_width_f_bins = inj_stats.detected_bin_midpoints_fluence[1]
        detected_det_frac_fluence = inj_stats.detected_det_frac_fluence
        detected_det_frac_fluence = pd.DataFrame(detected_det_frac_fluence)
        detected_det_frac_fluence.interpolate(
            limit_direction="both", axis=0, inplace=True
        )
        # convert back to numpy array
        detected_det_frac_fluence = detected_det_frac_fluence.to_numpy()
        self.detected_interp_fluence = scipy.interpolate.RegularGridInterpolator(
            (detected_fluence_bins, detected_width_f_bins),
            detected_det_frac_fluence,
            bounds_error=False,
            fill_value=None,
        )

        injected_snr = inj_stats.unique_snrs
        injected_width = inj_stats.unique_widths
        injected_det_frac = inj_stats.det_frac_matrix_snr
        self.injected_interp_snr = scipy.interpolate.RegularGridInterpolator(
            (injected_snr, injected_width),
            injected_det_frac,
            bounds_error=False,
            fill_value=None,
        )

        self.detected_snr_bins = detected_snr_bins
        self.detected_width_bins = detected_width_bins
        self.detected_det_frac_snr = detected_det_frac_snr

        self.detected_fluence_bins = detected_fluence_bins
        self.detected_width_f_bins = detected_width_f_bins
        self.detected_det_frac_fluence = detected_det_frac_fluence

        self.injected_snr = injected_snr
        self.injected_width = injected_width
        self.injected_det_frac = injected_det_frac

        self.cupy_detected_det_frac_snr = cp.asarray(detected_det_frac_snr_stage1)
        self.cupy_detected_snr_bins = cp.asarray(detected_snr_bins_stage1)
        self.cupy_detected_width_bins = cp.asarray(detected_width_bins_stage1)

        self.cupy_detected_det_frac_fluence = cp.asarray(detected_det_frac_fluence)
        self.cupy_detected_fluence_bins = cp.asarray(detected_fluence_bins)
        self.cupy_detected_width_f_bins = cp.asarray(detected_width_f_bins)

        self.cupy_injected_det_frac = cp.asarray(injected_det_frac)
        self.cupy_injected_snr = cp.asarray(injected_snr)
        self.cupy_injected_width = cp.asarray(injected_width)

        self.cupy_detected_interp_snr = cpinterp.RegularGridInterpolator(
            (self.cupy_detected_snr_bins, self.cupy_detected_width_bins),
            self.cupy_detected_det_frac_snr,
            bounds_error=False,
            fill_value=None,
        )
        self.cupy_detected_interp_fluence = cpinterp.RegularGridInterpolator(
            (self.cupy_detected_fluence_bins, self.cupy_detected_width_f_bins),
            self.cupy_detected_det_frac_fluence,
            bounds_error=False,
            fill_value=None,
        )
        self.cupy_injected_interp_snr = cpinterp.RegularGridInterpolator(
            (self.cupy_injected_snr, self.cupy_injected_width),
            self.cupy_injected_det_frac,
            bounds_error=False,
            fill_value=None,
        )

        snr_arr = np.linspace(0, 50, 1000)
        width_arr = np.linspace(1, 60, 1000)*1e-3
        fluence_arr = np.linspace(0, 1, 1000)
        snr_grid, width_grid = np.meshgrid(snr_arr, width_arr, indexing="ij")
        points = (snr_grid, width_grid)
        interp_res_snr = self.p_detect_cpu(points, fluence=False)
        fluence_grid, width_grid = np.meshgrid(fluence_arr, width_arr, indexing="ij")
        points = (fluence_grid, width_grid)
        interp_res_fluence = self.p_detect_cpu(points, fluence=True)

        # detfn = p_detect(snr_arr,min_snr_cutoff=min_snr_cutoff,flux_cal=flux_cal)
        # get the snr cutoff by finding when detfn is larger than 0.05
        # snr_cutoff = inj_stats._snr[np.where(detfn>0)[0][0]]
        if plot:
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            mesh = ax[0].pcolormesh(width_grid * 1e3, snr_arr, interp_res_snr)
            ax[0].set_xlabel("width")
            ax[0].set_ylabel("snr")
            cbar = plt.colorbar(mesh, ax=ax[0])
            cbar.set_label("detection fraction")
            mesh = ax[1].pcolormesh(
                detected_width_bins * 1e3, detected_snr_bins, detected_det_frac_snr
            )
            ax[1].set_xlabel("width")
            ax[1].set_ylabel("snr")
            cbar = plt.colorbar(mesh, ax=ax[1])
            cbar.set_label("detection fraction")

            # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            # mesh = ax[0].pcolormesh(width_grid * 1e3, fluence_arr, interp_res_fluence)
            # ax[0].set_xlabel("width")
            # ax[0].set_ylabel("fluence")
            # cbar = plt.colorbar(mesh, ax=ax[0])
            # cbar.set_label("detection fraction")
            # mesh = ax[1].pcolormesh(
            #     detected_width_f_bins * 1e3,
            #     detected_fluence_bins,
            #     inj_stats.detected_det_frac_fluence,
            # )
            # ax[1].set_xlabel("width")
            # ax[1].set_ylabel("fluence")
            # cbar = plt.colorbar(mesh, ax=ax[1])
            # cbar.set_label("detection fraction")
            plt.show()

        # if snr_cutoff < 1.3:
        #     print("WARNING SNR CUTOFF IS LESS THAN 1.3")
        #     snr_cutoff = 1.3
        # assign the errors in the different directions
        self.detected_error_snr = inj_stats.detect_error_snr
        self.detected_error_width = inj_stats.detect_error_width
        self.detected_error_fluence = inj_stats.detect_error_fluence
        return snr_cutoff, width_cutoff

    def logistic(self, x, k, x0):
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

    def n_detect(self, snr_emit):
        # snr emit is the snr that the emitted pulse has
        p = p_detect(snr_emit)
        # simulate random numbers between 0 and 1
        rands = np.random.rand(len(p))
        # probability the random number is less than p gives you an idea of what will be detected
        detected = snr_emit[rands < p]
        return detected
