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
from scipy.stats import multivariate_normal
# from statistics_basic import load_detection_fn, p_detect_cupy, p_detect_cpu
global det_error
# det_error = statistics_basic.det_error
# print("det_error for LN",det_error)
import time

###############################CUPY FUNCTIONS##################################
import cupy as cp

def multivar_norm(x, mu, sigma, corr):
    # x is a tuple of (snr,width)
    # mu is a tuple of (mu_amp,mu_w)
    # sigma is a tuple of (sigma_amp,sigma_w)
    # corr is the correlation between the two
    # lower and upper cutoff parameters added
    # import pdb; pdb.set_trace()
    pdf = cp.exp(
        -(
            (x[0] - mu[0]) ** 2 / sigma[0] ** 2
            + (x[1] - mu[1]) ** 2 / sigma[1] ** 2
            - 2 * corr * (x[0] - mu[0]) * (x[1] - mu[1])
            / (sigma[0] * sigma[1])
        )
        / (2 * (1 - corr ** 2))
    ) / (
        2
        * cp.pi
        * sigma[0]
        * sigma[1]
        * cp.sqrt(1 - corr ** 2)
    )
    return pdf

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



class statistics_ln(sb):
    def calculate_pdet(self, amp, width):
        amp = cp.array(amp)
        width = cp.array(width)
        p_det = self.p_detect_cupy((amp, width))
        self.p_det_cupy = p_det
        self.p_det = p_det.get()

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
    ):
        sigma_lim = 5
        true_upper = mu + (sigma_lim * std)
        true_lower = mu - (sigma_lim * std)
        true_amp_array = cp.linspace(true_lower, true_upper, 1000)
        true_upper = mu_w + (sigma_lim * std_w)
        true_lower = mu_w - (sigma_lim * std_w)
        true_width_array = cp.linspace(true_lower, true_upper, 1001)
        true_dist_amp = gaussian_cupy(true_amp_array, mu, std)
        true_dist_width = gaussian_cupy(true_width_array, mu_w, std_w)
        points = (cp.exp(true_amp_array[:, cp.newaxis]), cp.exp(true_width_array[cp.newaxis, :]))
        pdet = self.p_detect_cpu_true_cupy(points)
        # plt.figure()
        # plt.pcolormesh( cp.exp(true_amp_array).get(),cp.exp(true_width_array).get(), pdet.get().T)
        # plt.colorbar()
        # plt.show()
        p_ndet_st_wt = pdet*true_dist_amp[:, cp.newaxis]*true_dist_width[cp.newaxis, :]


        p_ndet = 1 - cp.trapz(
            cp.trapz(p_ndet_st_wt, true_amp_array, axis=0), true_width_array
        )
        likelihood = cp.log(p_ndet) * (N - n)
        # print(p_ndet)
        return likelihood, p_ndet

    def second(
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
    ):
        sigma_lim = 4
        true_upper = mu + (sigma_lim * std)
        true_lower = mu - (sigma_lim * std)
        true_amp_array = np.linspace(true_lower, true_upper, 1000)
        true_upper = mu_w + (sigma_lim * std_w)
        true_lower = mu_w - (sigma_lim * std_w)
        true_width_array = np.linspace(true_lower, true_upper, 1001)

        true_dist_amp = norm.pdf(true_amp_array, mu, std)
        true_dist_width = norm.pdf(true_width_array, mu_w, std_w)
        points = (np.exp(true_amp_array[:, np.newaxis]), np.exp(true_width_array[np.newaxis, :]))
        pndet = (1 - self.p_detect_cpu_true(points))
        # plt.figure()
        # plt.pcolormesh(np.exp(true_width_array).get(), np.exp(true_amp_array).get(), pndet.get())
        # plt.colorbar()
        # plt.show()
        p_ndet_st_wt = pndet*true_dist_amp[:, np.newaxis]*true_dist_width[np.newaxis, :]


        p_ndet = np.trapz(
            np.trapz(p_ndet_st_wt, true_amp_array, axis=0), true_width_array
        )
        likelihood = np.log(p_ndet) * (N - n)
        return likelihood, p_ndet


    def first_cupy(
        self,
        amp,
        width,
        mu,
        std,
        mu_w,
        std_w,
        corr,
        sigma_amp=0.4,
        sigma_w=1e-3,
        a=0,
        lower_c=0,
        upper_c=np.inf,
    ):
        x_len = 101
        x_len_w = 102
        amp = cp.array(amp)
        width = cp.array(width)
        # amp is the detected amps
        # width is the detected widths
        # make an array of lower and upper limits for the true_log amp array
        sigma_lim = 5
        true_lower = amp - sigma_lim * sigma_amp
        true_lower[true_lower < 0] = cp.exp(-20)
        true_upper = amp + sigma_lim * sigma_amp

        true_lower_w = width - sigma_lim * sigma_w
        true_lower_w[true_lower_w < 0] = cp.exp(-20)
        true_upper_w = width + sigma_lim * sigma_w
        # generate a mesh of amps
        # true_amp_mesh = cp.zeros((len(amp), x_len))
        # for i, (l, u) in enumerate(zip(true_lower, true_upper)):
            # true_amp_mesh[i, :] = cp.linspace(cp.log(l), cp.log(u), x_len)
        true_amp_mesh = cp.linspace(cp.log(true_lower),cp.log(true_upper),x_len).T
        true_amp_mesh = true_amp_mesh[:,:,cp.newaxis]
        true_w_mesh = cp.linspace(cp.log(true_lower_w),cp.log(true_upper_w),x_len_w).T
        true_w_mesh = true_w_mesh[:,cp.newaxis,:]
        amp = amp[:, cp.newaxis, cp.newaxis]
        width = width[:, cp.newaxis, cp.newaxis]
        gaussian_error_amp = gaussian_cupy(amp, cp.exp(true_amp_mesh), sigma_amp)
        gaussian_error_w = gaussian_cupy(width, cp.exp(true_w_mesh), sigma_w)
        pos = (true_amp_mesh, true_w_mesh)
        lognorm_amp_dist = multivar_norm(pos, [mu,mu_w], [std,std_w], corr)
        mult = gaussian_error_amp * lognorm_amp_dist * gaussian_error_w
        # integral over true_amp_mesh
        # integral_amp_only = cp.zeros((len(amp), x_len_w))
        # for i in range(len(amp)):
            # integral_amp_only[i, :] = cp.trapz(mult[i, :, :], true_amp_mesh[i, :, :], axis=0)

        integral_amp_only = cp.trapz(mult, true_amp_mesh, axis=1)
        integral = cp.trapz(integral_amp_only, true_w_mesh[:,0,:], axis=1)
        amp = amp[:, 0, 0]
        width = width[:, 0, 0]
        likelihood = integral * self.p_det_cupy
        return cp.sum(cp.log(likelihood)), self.p_det_cupy


    def first(
        self,
        amp,
        width,
        mu,
        std,
        mu_w,
        std_w,
        sigma_amp=0.4,
        sigma_w=1e-3,
        a=0,
        lower_c=0,
        upper_c=np.inf,
    ):
        x_len = 1000
        # amp is the detected amps
        # width is the detected widths
        # make an array of lower and upper limits for the true_log amp array
        sigma_lim = 8
        true_lower = amp - sigma_lim * sigma_amp
        true_lower[true_lower < 0] = np.exp(-20)
        true_upper = amp + sigma_lim * sigma_amp
        # generate a mesh of amps
        # true_amp_mesh = np.zeros((len(amp), x_len))
        true_amp_mesh = np.linspace(np.log(true_lower),np.log(true_upper),x_len).T

        # for i, (l, u) in enumerate(zip(true_lower, true_upper)):
            # true_amp_mesh[i, :] = np.linspace(np.log(l), np.log(u), x_len)

        amp = amp[:, np.newaxis]
        gaussian_error_amp = norm.pdf(amp, np.exp(true_amp_mesh), sigma_amp)
        lognorm_amp_dist = norm.pdf(true_amp_mesh, mu, std)
        mult_amp = gaussian_error_amp * lognorm_amp_dist
        # integral over true_amp_mesh
        integral_amp = np.trapz(mult_amp, true_amp_mesh, axis=1)
        # now do the same for width
        x_len = 1001
        true_lower_w = width - sigma_lim * sigma_w
        true_lower_w[true_lower_w < 0] = np.exp(-20)
        true_upper_w = width + sigma_lim * sigma_w
        # generate a mesh of widths
        true_w_mesh = np.linspace(np.log(true_lower_w),np.log(true_upper_w),x_len).T
        # true_w_mesh = np.zeros((len(width), x_len))
        # for i, (l, u) in enumerate(zip(true_lower_w, true_upper_w)):
            # true_w_mesh[i, :] = np.linspace(np.log(l), np.log(u), x_len)
        width = width[:, np.newaxis]
        gaussian_error_w = norm.pdf(width, np.exp(true_w_mesh), sigma_w)
        lognorm_w_dist = norm.pdf(true_w_mesh, mu_w, std_w)
        mult_w = gaussian_error_w * lognorm_w_dist
        # integral over true_w_mesh
        integral_w = np.trapz(mult_w, true_w_mesh, axis=1)
        # #multiple against each other
        # convolve the two arrays
        # interpolate the values for amp
        points = (amp[:,0],width[:,0])
        likelihood = integral_amp * integral_w * self.p_det
        # plt.figure()
        # plt.scatter(width.get(),amp.get(),s=10, c=self.p_det.get())
        # #set colorbar range from 0 - 1
        # plt.clim(0,1)
        # plt.xlabel("Amplitude")
        # plt.ylabel("Width")

        # plt.figure()
        # plt.scatter(amp.get(), width.get(), s=10, c=likelihood.get())
        # plt.xlabel("Amplitude")
        # plt.ylabel("Width")
        # plt.colorbar()
        # plt.show()
        return np.sum(np.log(likelihood)), self.p_det


    #################CUPY END#####################

    def first_plot(
        self,
        amp,
        width,
        mu,
        std,
        mu_w,
        std_w,
        corr,
        sigma_amp=0.4,
        sigma_w=1e-3,
        a=0,
        lower_c=0,
        upper_c=np.inf,
    ):
        x_len = 101
        x_len_w = 102
        amp = cp.array(amp)
        width = cp.array(width)
        # amp is the detected amps
        # width is the detected widths
        # make an array of lower and upper limits for the true_log amp array
        sigma_lim = 5
        true_lower = amp - sigma_lim * sigma_amp
        true_lower[true_lower < 0] = cp.exp(-20)
        true_upper = amp + sigma_lim * sigma_amp

        true_lower_w = width - sigma_lim * sigma_w
        true_lower_w[true_lower_w < 0] = cp.exp(-20)
        true_upper_w = width + sigma_lim * sigma_w
        # generate a mesh of amps
        # true_amp_mesh = cp.zeros((len(amp), x_len))
        # for i, (l, u) in enumerate(zip(true_lower, true_upper)):
            # true_amp_mesh[i, :] = cp.linspace(cp.log(l), cp.log(u), x_len)
        true_amp_mesh = cp.linspace(cp.log(true_lower),cp.log(true_upper),x_len).T
        true_amp_mesh = true_amp_mesh[:,:,cp.newaxis]
        true_w_mesh = cp.linspace(cp.log(true_lower_w),cp.log(true_upper_w),x_len_w).T
        true_w_mesh = true_w_mesh[:,cp.newaxis,:]
        amp = amp[:, cp.newaxis, cp.newaxis]
        width = width[:, cp.newaxis, cp.newaxis]
        gaussian_error_amp = gaussian_cupy(amp, cp.exp(true_amp_mesh), sigma_amp)
        gaussian_error_w = gaussian_cupy(width, cp.exp(true_w_mesh), sigma_w)
        pos = (true_amp_mesh, true_w_mesh)
        lognorm_amp_dist = multivar_norm(pos, [mu,mu_w], [std,std_w], corr)
        mult = gaussian_error_amp * lognorm_amp_dist * gaussian_error_w
        # integral over true_amp_mesh
        # integral_amp_only = cp.zeros((len(amp), x_len_w))
        # for i in range(len(amp)):
            # integral_amp_only[i, :] = cp.trapz(mult[i, :, :], true_amp_mesh[i, :, :], axis=0)

        integral_amp_only = cp.trapz(mult, true_amp_mesh, axis=1)
        integral = cp.trapz(integral_amp_only, true_w_mesh[:,0,:], axis=1)
        amp = amp[:, 0, 0]
        width = width[:, 0, 0]
        p_det = self.p_detect_cupy((amp, width))
        likelihood = integral * p_det
        return likelihood.get(), p_det.get()

    def total_p(
        self,
        X,
        snr_arr=None,
        width_arr=None,
        use_a=False,
        use_cutoff=True,
        cuda_device=0,
    ):
    # print("starting loglike")
        start = time.time()
        snr_arr = np.array(snr_arr)
        width_arr = np.array(width_arr)
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
        sigma_snr = self.detected_error_snr
        sigma_width = self.detected_error_width
        n = len(snr_arr)
        f, p_det_f = self.first(
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
        )
        first_time = time.time()
        # print("finished f")
        if np.isnan(f):
            print("f is nan")
            return -np.inf
        # s = second(len(snr_arr), mu, std, N, sigma_snr=sigma_snr)
        s, p_det_s = self.second(
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
        )
        second_time = time.time()
        # print("finished s")
        if np.isnan(s):
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
        # print(f"f: {f}, s: {s}, log_NCn: {log_NCn} loglike: {loglike}")
        return loglike

    def total_p_cupy(
        self,
        X,
        snr_arr=None,
        width_arr=None,
        use_a=False,
        use_cutoff=True,
        cuda_device=0,
    ):
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
            sigma_snr = self.detected_error_snr
            sigma_width = self.detected_error_width
            n = len(snr_arr)
            f, p_det_f = self.first_cupy(
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
            )
            first_time = time.time()
            # print("finished f")
            if cp.isnan(f):
                print("f is nan")
                return -np.inf
            # s = second(len(snr_arr), mu, std, N, sigma_snr=sigma_snr)
            s, p_det_s = self.second_cupy(
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
            # print(f"f: {f}, s: {s}, log_NCn: {log_NCn} loglike: {loglike}")
        return loglike.get()

    def negative_loglike(self, X, det_snr):
        x = {"mu": X[0], "std": X[1], "N": X[2], "snr_arr": det_snr}
        return -1 * total_p(x)

    def mean_var_to_mu_std(self, mean, var):
        mu = np.log(mean**2 / np.sqrt(var + mean**2))
        std = np.sqrt(np.log(var / mean**2 + 1))
        return mu, std

    def mu_std_to_mean_var(self, mu, std):
        mean = np.exp(mu + std**2 / 2)
        var = (np.exp(std**2) - 1) * np.exp(2 * mu + std**2)
        return mean, var

    def likelihood_lognorm(self, mu_arr, std_arr, N_arr, det_snr, mesh_size=20):
        # # create a mesh grid of N, mu and stds
        mat = np.zeros((len(mu_arr), len(std_arr), len(N_arr)))
        if max(det_snr) > 100:
            xlim = max(det_snr) * 2
        else:
            xlim = 100
        # with Pool(2) as po:
        X = []
        Y = []
        with cp.cuda.Device(0):
            det_snr = cp.array(det_snr)
        for i, mu_i in enumerate(mu_arr):
            for j, std_i in enumerate(std_arr):
                for k, N_i in enumerate(N_arr):
                    # change the mu to a different definition
                    mean_i, var_i = mu_std_to_mean_var(mu_i, std_i)
                    upper_c = mean_i * 50
                    X.append(
                        {
                            "mu": mu_i,
                            "std": std_i,
                            "N": N_i,
                            "snr_arr": det_snr,
                            "lower_c": 0,
                            "upper_c": upper_c,
                        }
                    )
                    Y.append([mu_i, std_i, N_i])
        Y = np.array(Y)
        # m = np.array(po.map(total_p, X))
        m = []
        for ind, v in enumerate(X):
            print(f"{ind}/{len(X)}")
            m.append(total_p(v, det_snr, use_cutoff=True, cuda_device=0, xlim=xlim))
        m = np.array(m)
        for i, mu_i in enumerate(mu_arr):
            for j, std_i in enumerate(std_arr):
                for k, N_i in enumerate(N_arr):
                    ind = np.sum((Y == [mu_i, std_i, N_i]), axis=1) == 3
                    mat[i, j, k] = m[ind]

        return mat
