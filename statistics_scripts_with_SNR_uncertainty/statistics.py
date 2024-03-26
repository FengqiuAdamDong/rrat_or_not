#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from math import comb
from scipy.special import gammaln
from multiprocessing import Pool
import os
import dill
import scipy
from cupyx.scipy.special import gammaln as cupy_gammaln
from cupyx.scipy.special import erf as cupy_erf
from statistics_basic import statistics_basic as sb

# from statistics_basic import load_detection_fn, p_detect_cupy, p_detect_cpu
global det_error
# det_error = statistics_basic.det_error
# print("det_error for LN",det_error)
import time

###############################CUPY FUNCTIONS##################################
import cupy as cp


def lognorm_dist_cupy(x, mu, sigma, lower_c=0, upper_c=cp.inf):
    # lower and upper cutoff parameters added
    pdf = cp.zeros(x.shape)
    mask = (x > lower_c) & (x < upper_c)
    pdf[mask] = cp.exp(-((cp.log(x[mask]) - mu) ** 2) / (2 * sigma**2)) / (
        (x[mask]) * sigma * cp.sqrt(2 * cp.pi)
    )

    def argument(c, mu, sigma):
        if c == 0:
            return -cp.inf
        return (cp.log(c) - mu) / (sigma * cp.sqrt(2))

    pdf = (
        2
        * pdf
        / (
            cupy_erf(argument(upper_c, mu, sigma))
            - cupy_erf(argument(lower_c, mu, sigma))
        )
    )
    return pdf


def gaussian_cupy(x, mu, sigma):
    return cp.exp(-((x - mu) ** 2) / (2 * sigma**2)) / (sigma * cp.sqrt(2 * cp.pi))


def lognorm_dist(x, mu, sigma, lower_c=0, upper_c=np.inf):
    # lower and upper cutoff parameters added
    pdf = np.zeros(x.shape)
    mask = (x > lower_c) & (x < upper_c)
    pdf[mask] = np.exp(-((np.log(x[mask]) - mu) ** 2) / (2 * sigma**2)) / (
        (x[mask]) * sigma * np.sqrt(2 * np.pi)
    )

    def argument(c, mu, sigma):
        if c == 0:
            return -np.inf
        return (np.log(c) - mu) / (sigma * np.sqrt(2))

    pdf = (
        2
        * pdf
        / (
            scipy.special.erf(argument(upper_c, mu, sigma))
            - scipy.special.erf(argument(lower_c, mu, sigma))
        )
    )
    return pdf

def exponential_dist_cupy(x, k):
    pdf = k * cp.exp(-k * x)
    return pdf

def exponential_dist_lnx(x, k):
    pdf = k * cp.exp(x-k*cp.exp(x))
    return pdf


class statistics_ln(sb):
    def calculate_pdet(self, amp, width):
        amp = cp.array(amp)
        width = cp.array(width)
        p_det = self.p_detect_cupy((amp, width))

        self.p_det_unfiltered = p_det
        self.p_det_cupy = self.p_det_unfiltered[self.p_det_unfiltered > 0]

        self.p_det = self.p_det_cupy.get()


    def second_cupy(
        self,
        n,
        mu,
        std,
        mu_w,
        std_w,
        N,
        sigma_amp=0.4,
        sigma_w=1e-3,
        a=0,
        lower_c=0,
        upper_c=np.inf,
        amp_dist="ln",
        w_dist = "ln",
    ):
        sigma_lim = 5
        xlen_amp = 1000
        xlen_width = 1001

        if amp_dist == "ln":
            true_upper = mu + (sigma_lim * std)
            true_lower = mu - (sigma_lim * std)
            true_amp_array = cp.linspace(true_lower, true_upper, xlen_amp)
            true_dist_amp = gaussian_cupy(true_amp_array, mu, std)
            points_amp = cp.exp(true_amp_array[:, cp.newaxis])

        elif amp_dist == "exp":
            true_lower = cp.log(1/mu)-6
            true_upper = cp.log(1/mu)+2.5
            true_amp_array = cp.linspace(true_lower, true_upper, xlen_amp)
            true_dist_amp = exponential_dist_lnx(true_amp_array, mu)
            points_amp = cp.exp(true_amp_array[:,cp.newaxis])

        if w_dist == "ln":
            true_upper = mu_w + (sigma_lim * std_w)
            true_lower = mu_w - (sigma_lim * std_w)
            true_width_array = cp.linspace(true_lower, true_upper, xlen_width)
            true_dist_width = gaussian_cupy(true_width_array, mu_w, std_w)
            points_width = cp.exp(true_width_array[cp.newaxis, :])
        elif w_dist == "exp":
            #in the exp case, mu_w is the rate parameter
            true_lower = cp.log(1/mu_w)-6
            true_upper = cp.log(1/mu_w)+2.5
            true_width_array = cp.linspace(true_lower, true_upper, xlen_width)
            true_dist_width = exponential_dist_lnx(true_width_array, mu_w)
            points_width = cp.exp(true_width_array[cp.newaxis, :])

        points = (points_amp, points_width)
        pdet = self.p_detect_cpu_true_cupy(points)

        # plt.figure()
        # plt.pcolormesh( cp.exp(true_amp_array).get(),cp.exp(true_width_array).get(), pdet.get().T)
        # plt.colorbar()
        # plt.show()
        p_ndet_st_wt = pdet*true_dist_amp[:, cp.newaxis]*true_dist_width[cp.newaxis, :]


        p_ndet = 1 - cp.trapz(
            cp.trapz(p_ndet_st_wt, true_amp_array, axis=0), true_width_array
        )
        loglike = cp.log(p_ndet) * (N - n)
        # print(p_ndet)
        return  loglike, p_ndet

    def first_cupy(
        self,
        amp, #detected amps
        width, #detected widths
        mu,
        std,
        mu_w,
        std_w,
        sigma_amp,
        sigma_w,
        a=0,
        lower_c=0,
        upper_c=np.inf,
        amp_dist="ln",
        w_dist = "ln",
    ):
        x_len = 1000
        # amp is the detected amps
        # width is the detected widths
        # make an array of lower and upper limits for the true_log amp array
        #
        sigma_lim = 8
        true_lower_gauss = amp - sigma_lim * sigma_amp
        true_upper_gauss = amp + sigma_lim * sigma_amp
        if amp_dist == "ln":
            true_upper_ln = cp.exp(mu + (std * 5))
            true_lower_ln = cp.maximum(cp.exp(mu - (std * 5)), 1e-20)
            true_lower = cp.maximum(true_lower_gauss, true_lower_ln)
            true_upper = cp.minimum(true_upper_ln, true_upper_gauss)
            if (true_lower > true_upper).any():
                return -cp.inf, cp.nan
            # generate a mesh of amps

            amp = amp[:, cp.newaxis]
            true_amp_mesh = cp.linspace(cp.log(true_lower),cp.log(true_upper),x_len).T
            gaussian_error_amp = gaussian_cupy(amp, cp.exp(true_amp_mesh), sigma_amp)
            lognorm_amp_dist = gaussian_cupy(true_amp_mesh, mu, std)
            amp_dist = lognorm_amp_dist

        elif amp_dist == "exp":
            true_lower_exp = (1/mu)*cp.exp(-6)
            true_upper_exp = (1/mu)*cp.exp(2.5)
            true_lower = cp.maximum(true_lower_gauss, true_lower_exp)
            true_upper = cp.minimum(true_upper_gauss, true_upper_exp)
            if (true_lower > true_upper).any():
                return -cp.inf, cp.nan

            amp = amp[:, cp.newaxis]
            true_amp_mesh = cp.linspace(cp.log(true_lower), cp.log(true_upper), x_len).T
            gaussian_error_amp = gaussian_cupy(amp, cp.exp(true_amp_mesh), sigma_amp)
            exponential_amp_dist = exponential_dist_lnx(true_amp_mesh, mu)
            amp_dist = exponential_amp_dist

        #integrate over the true amplitude
        mult_amp = gaussian_error_amp * amp_dist
        integral_amp = cp.trapz(mult_amp, true_amp_mesh, axis=1)

        # now do the same for width
        x_len = 1001
        true_lower_w_gauss = width - sigma_lim * sigma_w
        true_upper_w_gauss = width + sigma_lim * sigma_w
        if w_dist == "ln":

            true_upper_w_alt = cp.exp(mu_w + std_w*5)
            true_lower_w_alt = np.maximum(cp.exp(mu_w - std_w*5),1e-20)

            true_lower_w = cp.maximum(true_lower_w_alt, true_lower_w_gauss)
            true_upper_w = cp.minimum(true_upper_w_gauss, true_upper_w_alt)
            # generate a mesh of widths
            if (true_lower_w > true_upper_w).any():
                return -cp.inf, cp.nan

            width = width[:, cp.newaxis]
            true_w_mesh = cp.linspace(cp.log(true_lower_w),cp.log(true_upper_w),x_len).T
            lognorm_w_dist = gaussian_cupy(true_w_mesh, mu_w, std_w)
            w_dist = lognorm_w_dist
            gaussian_error_w = gaussian_cupy(width, cp.exp(true_w_mesh), sigma_w)

        elif w_dist == "exp":
            true_lower_w_alt = (1/mu_w)*cp.exp(-6)
            true_upper_w_alt = (1/mu_w)*cp.exp(2.5)
            true_lower_w = cp.maximum(true_lower_w_gauss, true_lower_w_alt)
            true_upper_w = cp.minimum(true_upper_w_gauss, true_upper_w_alt)
            if (true_lower_w > true_upper_w).any():
                return -cp.inf, cp.nan

            width = width[:, cp.newaxis]
            true_w_mesh = cp.linspace(cp.log(true_lower_w),cp.log(true_upper_w),x_len).T
            gaussian_error_w = gaussian_cupy(width, cp.exp(true_w_mesh), sigma_w)
            exp_w_dist = exponential_dist_lnx(true_w_mesh, mu_w)
            w_dist = exp_w_dist
        mult_w = gaussian_error_w * w_dist
        # integral over true_w_mesh
        integral_w = cp.trapz(mult_w, true_w_mesh, axis=1)

        #get the likelihood
        #
        likelihood = integral_amp * integral_w * self.p_det_cupy
        # probably don't need this line of debug
        # likelihood = likelihood[self.p_det_cupy > 0]
        loglike = cp.log(likelihood)
        # print(f"amp = {amp[indmin]}, width = {width[indmin]} likelihood = {likelihood[indmin]}")
        if cp.isnan(loglike).any():
            import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        return cp.sum(loglike), loglike

    #################CUPY END#####################

    def total_p_cupy(
        self,
        X,
        snr_arr=None,
        width_arr=None,
        use_a=False,
        use_cutoff=True,
        low_width=False,
        cuda_device=0,
        amp_dist="ln",
        w_dist="ln",
    ):
        if amp_dist == "exp":
            if X["mu"] < 0:
                return -np.inf
        if w_dist == "exp":
            if X["mu_w"] < 0:
                return -np.inf
        # print("starting loglike")
        with cp.cuda.Device(cuda_device):
            start = time.time()
            snr_arr = cp.array(snr_arr)
            width_arr = cp.array(width_arr)
            transfer_time = time.time()
            mu = X["mu"]
            std = X["std"]
            mu_w = X["mu_w"]
            std_w = X["std_w"]
            N = X["N"]
            if use_a:
                a = X["a"]
            else:
                a = 0
            if use_cutoff:
                lower_c = X["lower_c"]
                upper_c = X["upper_c"]
            else:
                lower_c = 0
                upper_c = np.inf
            if lower_c > upper_c:
                print("lower_c is greater than upper_c")
                return -np.inf
            if snr_arr is None:
                snr_arr = X["snr_arr"]
                width_arr = X["width_arr"]
            if N < len(snr_arr):
                raise Exception(" N<n")

            # print(f"mu: {mu}, std: {std}, N: {N}, a: {a}, lower_c: {lower_c}, upper_c: {upper_c}")
            if low_width:
                # print("using low width error")
                sigma_snr = self.detected_error_snr_low_width
                sigma_width = self.detected_error_width_low_width
            else:
                sigma_snr = self.detected_error_snr
                sigma_width = self.detected_error_width

            n = len(snr_arr)
            f, _ = self.first_cupy(
                snr_arr,
                width_arr,
                mu,
                std,
                mu_w,
                std_w,
                sigma_amp=sigma_snr,
                sigma_w=sigma_width,
                a=a,
                lower_c=lower_c,
                upper_c=upper_c,
                amp_dist=amp_dist,
                w_dist=w_dist,
            )
            first_time = time.time()
            # print("finished f")
            if cp.isnan(f):
                print("f is nan")
                return -np.inf
            # s = second(len(snr_arr), mu, std, N, sigma_snr=sigma_snr)
            s, _ = self.second_cupy(
                n,
                mu,
                std,
                mu_w,
                std_w,
                N,
                sigma_amp=sigma_snr,
                sigma_w=sigma_width,
                a=a,
                lower_c=lower_c,
                upper_c=upper_c,
                amp_dist=amp_dist,
                w_dist=w_dist,
            )
            second_time = time.time()
            # print("finished s")
            if cp.isnan(s):
                print("s is nan")
                return -np.inf
            log_NCn = (
                cupy_gammaln(N + 1) - cupy_gammaln(n + 1) - cupy_gammaln(N - n + 1)
            )
            # print("finished log_NCn")
            loglike = f + s + log_NCn
            # loglike = np.array(loglike.get())
            overall_time = time.time()
            # print(
                # f"transfer time: {transfer_time-start}, f time: {first_time-transfer_time}, s time: {second_time-first_time}, overall time: {overall_time-start}"
            # )
            # print(f"f: {f}, s: {s}, log_NCn: {log_NCn} loglike: {loglike} mu: {mu}, std: {std}, N: {N}, mu_w: {mu_w}, std_w: {std_w}")
        return loglike.get()

