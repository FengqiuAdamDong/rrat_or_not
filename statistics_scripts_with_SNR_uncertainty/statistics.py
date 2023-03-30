#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from math import comb
from scipy.special import gammaln
from multiprocessing import Pool
import os
import scipy.optimize as opt
import dill

with open("inj_stats_fitted.dill", "rb") as inf:
    inj_stats = dill.load(inf)
popt = inj_stats.fit_logistic_amp
det_error = inj_stats.detect_error_amp

snr_arr = np.linspace(-2, 2, 1000)
print("det error", det_error)


def lognorm_dist(x, mu, sigma):
    pdf = np.zeros(x.shape)
    pdf[x > 0] = np.exp(-((np.log(x[x > 0]) - mu) ** 2) / (2 * sigma**2)) / (
        x[x > 0] * sigma * np.sqrt(2 * np.pi)
    )
    return pdf


def logistic(x, k, x0):
    L = 1
    return L / (1 + np.exp(-k * (x - x0)))


def p_detect(snr, cutoff=1):
    # this will just be an exponential rise at some center
    # added a decay rate variable just so things are compatible
    # load inj statistics
    k = popt[0]
    x0 = popt[1]
    # print(k,x0)
    L = 1
    detection_fn = np.zeros(len(snr))
    try:
        snr_limit = 1
        detection_fn[(snr > -snr_limit) & (snr < snr_limit)] = L / (
            1 + np.exp(-k * (snr[(snr > -snr_limit) & (snr < snr_limit)] - x0))
        )
        detection_fn[snr >= snr_limit] = 1
        detection_fn[snr <= -snr_limit] = 0
    except:
        import pdb

        pdb.set_trace()
    # detection_fn[snr<cutoff] = 0
    return detection_fn


detfn = p_detect(snr_arr)
plt.plot(snr_arr, detfn)


def n_detect(snr_emit):
    # snr emit is the snr that the emitted pulse has
    p = p_detect(snr_emit)
    # simulate random numbers between 0 and 1
    rands = np.random.rand(len(p))
    # probability the random number is less than p gives you an idea of what will be detected
    detected = snr_emit[rands < p]
    return detected


def first(snr, mu, std, sigma_snr):
    x_len = 10000
    const = 91
    xlim = np.exp(mu) * std * const
    x_lims = [-xlim, xlim]
    snr_true_array = np.linspace(x_lims[0], x_lims[1], x_len)
    expmodnorm = lognorm_dist(snr_true_array, mu, std)
    p_det = p_detect(snr_true_array)
    p_det_mod_norm = p_det * expmodnorm
    P_snr_true_giv_det = norm.pdf(snr_true_array, 0, sigma_snr)
    conv = np.convolve(p_det_mod_norm, P_snr_true_giv_det) * np.diff(snr_true_array)[0]
    conv_lims = [-(xlim * 2), xlim * 2]
    conv_snr_array = np.linspace(conv_lims[0], conv_lims[1], (x_len * 2) - 1)
    convolve_mu_snr = np.interp(snr, conv_snr_array, conv)
    try:
        log_convolve_mu_snr = np.zeros(len(convolve_mu_snr))
        log_convolve_mu_snr[convolve_mu_snr == 0] = -np.inf
        log_convolve_mu_snr[convolve_mu_snr > 0] = np.log(
            convolve_mu_snr[convolve_mu_snr > 0]
        )
    except:
        import pdb

        pdb.set_trace()
    return np.sum(log_convolve_mu_snr)


def second(n, mu, std, N, sigma_snr):

    x_len = 10000
    const = 91
    xlim = np.exp(mu) * std * const
    x_lims = [-xlim, xlim]
    snr = np.linspace(-xlim / 2, xlim / 2, 1000)

    snr_true_array = np.linspace(x_lims[0], x_lims[1], x_len)
    expmodnorm = lognorm_dist(snr_true_array, mu, std)
    p_det = 1 - p_detect(snr_true_array)
    p_det_mod_norm = p_det * expmodnorm

    P_snr_true_giv_det = norm.pdf(snr_true_array, 0, sigma_snr)
    conv = np.convolve(p_det_mod_norm, P_snr_true_giv_det) * np.diff(snr_true_array)[0]
    conv_lims = [-(xlim * 2), xlim * 2]
    conv_snr_array = np.linspace(conv_lims[0], conv_lims[1], (x_len * 2) - 1)
    convolve_mu_snr = np.interp(snr, conv_snr_array, conv)
    integral = np.trapz(convolve_mu_snr, snr)
    try:
        p_second_int = np.log(integral)
    except:
        import pdb

        pdb.set_trace()
    # plt.plot(snr_true_array,expmodnorm)
    # plt.show()
    if integral > 1:
        print("Integral error", integral)
        p_second_int = 1
        import pdb

        pdb.set_trace()
    return p_second_int * (N - n)


def total_p(X):
    mu = X["mu"]
    std = X["std"]
    N = X["N"]
    snr_arr = X["snr_arr"]
    if N < len(snr_arr):
        raise Exception(" N<n")
    sigma_snr = det_error
    f = first(snr_arr, mu, std, sigma_snr=sigma_snr)
    s = second(len(snr_arr), mu, std, N, sigma_snr=sigma_snr)
    n = len(snr_arr)
    log_NCn = gammaln(N + 1) - gammaln(n + 1) - gammaln(N - n + 1)
    return log_NCn + f + s


def negative_loglike(X, det_snr):
    x = {"mu": X[0], "std": X[1], "N": X[2], "snr_arr": det_snr}
    return -1 * total_p(x)


def likelihood_lognorm(mu_arr, std_arr, N_arr, det_snr, mesh_size=20):
    # # create a mesh grid of N, mu and stds
    mat = np.zeros((mesh_size, mesh_size + 1, mesh_size + 2))
    with Pool(50) as po:
        for i, mu_i in enumerate(mu_arr):
            for j, std_i in enumerate(std_arr):
                X = []
                for k, N_i in enumerate(N_arr):
                    X.append({"mu": mu_i, "std": std_i, "N": N_i, "snr_arr": det_snr})
                mat[i, j, :] = po.map(total_p, X)
                # for ind,v in enumerate(X):
                # mat[i,j,ind] = total_p(v)
    return mat


if __name__ == "__main__":
    from simulate_pulse import simulate_pulses
    from simulate_pulse import simulate_pulses_exp

    # x = np.linspace(0,5,100)
    # y = p_detect(x)
    # plt.plot(x,y)
    # plt.show()
    pos_array = []
    for a in range(1):
        obs_t = 500000
        mu = 0.3
        std = 0.1
        p = 2
        frac = 0.001
        pulse_snrs = simulate_pulses(obs_t, p, frac, mu, std)
        mesh_size = 50

        det_snr = n_detect(pulse_snrs)
        mu_arr = np.linspace(mu - 0.1, mu + 0.05, mesh_size)
        std_arr = np.linspace(std - 0.04, std + 0.05, mesh_size + 1)
        N_arr = np.linspace(
            (obs_t * frac / p) * 0.5, (obs_t * frac / p) * 2, mesh_size + 2, dtype=int
        )
        print(
            "number of generated pulses",
            len(pulse_snrs),
            "number of detections",
            len(det_snr),
        )

        # N_arr = np.linspace(len(det_snr),(obs_t/p)*frac*2,mesh_size+2,dtype=int)

        mat = likelihood_lognorm(mu_arr, std_arr, N_arr, det_snr, mesh_size)
        fn = f"d_{a}"
        print("saving", fn)
        save_dir = f"obs_{obs_t}_mu_{mu}_std_{std}_p_{p}_frac_{frac}"
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        fn = f"{save_dir}/{fn}"
        np.savez(
            fn,
            data=mat,
            mu=mu_arr,
            std=std_arr,
            N=N_arr,
            snrs=pulse_snrs,
            det=det_snr,
            true_mu=mu,
            true_std=std,
            p=p,
            true_frac=frac,
            obs_t=obs_t,
        )
        # import pdb; pdb.set_trace()
        # mat = mat-np.max(mat)
        mat = np.exp(mat)
        # integrate over mu and std
        posterior = np.trapz(np.trapz(mat, mu_arr, axis=0), std_arr, axis=0)
        pos_array.append(posterior)

        # np.save('simulated_pulses_0.65_0.1',[pulse_snrs,det_snr])
        # pulses = np.load('simulated_pulses_0.65_0.1.npy',allow_pickle=1)
        # pulse_snrs = pulses[0]
        # det_snr = pulses[1]

    np.save("posteriors", pos_array)
    plt.figure()
    plt.plot(N_arr, posterior)
    plt.xlabel("N")
    plt.title(f"# of simulated pulses:{len(pulse_snrs)} # of det pulses:{len(det_snr)}")
    plt.show()

    plt.figure()
    plt.hist(det_snr, bins=100)
    plt.title(f"total number of pulses:{len(det_snr)}")
    plt.xlabel(f"detected snr")
    plt.figure()
    plt.hist(pulse_snrs, bins=100)
    plt.xlabel("emmitted snr")
    plt.title(f"total number of pulses:{len(pulse_snrs)}")
    plt.show()
