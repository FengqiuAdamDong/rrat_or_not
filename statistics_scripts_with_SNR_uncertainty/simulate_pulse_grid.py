#!/usr/bin/env python3

import dill
import numpy as np
import sys
from simulate_pulse import simulate_pulses
from simulate_pulse import n_detect
from simulate_pulse import n_detect_true
from simulate_pulse import simulate_pulses_exp
from simulate_pulse import simulate_pulses_gauss
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import expon
from scipy.stats import truncnorm
import statistics_basic
import dill
import argparse
import os
import yaml


def simulate_and_process_data(
    detected_req,
    mode,
    mu_ln,
    std_ln,
    w_mu_ln,
    w_std_ln,
    lower,
    upper,
    sb,
    a,
    inj_file,
    dill_file,
    plot=True,
    out_fol="simulated_dir",
    snr_cutoff=2.0,
    width_cutoff=5e-3
):
    detected_pulses_snr = []
    detected_pulses_width = []
    detected_pulses_fluence = []
    detected_pulses_snr_true = []
    detected_pulses_width_true = []
    detected_pulses_fluence_true = []
    total_pulses_snr = []
    total_pulses_width = []
    total_pulses_fluence = []
    true_pulses_snr = []
    true_pulses_width = []
    true_pulses_fluence = []
    sigma_snr = sb.detected_error_snr
    sigma_width = sb.detected_error_width
    sigma_fluence = sb.detected_error_fluence

    while len(detected_pulses_snr) < detected_req:
        print(f"detected pulses: {len(detected_pulses_snr)}")
        obs_t = int(detected_req/10)
        p = 1
        f = 1
        if mode == "Exp":
            pulses = simulate_pulses_exp(
                obs_t, p, f, mu_ln, a, random=False, lower=lower, upper=upper
            )
        elif mode == "Lognorm":
            pulses = simulate_pulses(
                obs_t, p, f, mu_ln, std_ln, a, lower=lower, upper=upper, random=False
            )
        elif mode == "Gauss":
            pulses, a, b = simulate_pulses_gauss(
                obs_t, p, f, mu_ln, std_ln, random=True
            )
        rv_snr = norm(loc=0, scale=sigma_snr).rvs(len(pulses))
        d_pulses = rv_snr + pulses
        pulses_width = simulate_pulses(
            obs_t, p, f, w_mu_ln, w_std_ln, a, lower=0, upper=np.inf, random=False
        )
        rv_width = norm(loc=0, scale=sigma_width).rvs(len(pulses_width))
        d_pulses_width = rv_width + pulses_width

        # calculate the fluence from the width and snr, assume it is a gaussian pulse
        pulses_fluence = pulses_width * pulses / 0.3989
        rv_fluence = norm(loc=0, scale=sigma_fluence).rvs(len(pulses_fluence))
        d_pulses_fluence = rv_fluence + pulses_fluence

        d_snr, d_width, index = n_detect(d_pulses, d_pulses_width, sb)
        t_snr, t_width = n_detect_true(pulses, pulses_width, sb)
        if len(d_snr) > 0:
            detected_pulses_snr.extend(d_snr)
            detected_pulses_width.extend(d_width)
            detected_pulses_fluence.extend(d_pulses_fluence)
            detected_pulses_snr_true.extend(pulses[index])
            detected_pulses_width_true.extend(pulses_width[index])
            true_pulses_snr.extend(t_snr)
            true_pulses_width.extend(t_width)
        total_pulses_snr.extend(pulses)
        total_pulses_width.extend(pulses_width)
        total_pulses_fluence.extend(pulses_fluence)

    detected_pulses_snr = np.array(detected_pulses_snr).flatten()
    detected_pulses_width = np.array(detected_pulses_width).flatten()
    detected_pulses_fluence = np.array(detected_pulses_fluence).flatten()
    total_pulses_snr = np.array(total_pulses_snr).flatten()
    total_pulses_width = np.array(total_pulses_width).flatten()
    total_pulses_fluence = np.array(total_pulses_fluence).flatten()
    true_pulses_snr = np.array(true_pulses_snr).flatten()
    true_pulses_width = np.array(true_pulses_width).flatten()
    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].hist(
            detected_pulses_snr,
            bins="auto",
            density=True,
            label="fake data snr",
            alpha=0.5,
        )
        axes[0].hist(
            true_pulses_snr, bins="auto", density=True, label="true data snr", alpha=0.5
        )
        axes[0].hist(
            detected_pulses_snr_true,
            bins="auto",
            density=True,
            label="detected true data snr",
            alpha=0.5,
        )
        axes[1].hist(
            detected_pulses_width,
            bins="auto",
            density=True,
            label="fake data width",
            alpha=0.5,
        )
        axes[1].hist(
            true_pulses_width,
            bins="auto",
            density=True,
            label="true data width",
            alpha=0.5,
        )
        axes[1].hist(
            detected_pulses_width_true,
            bins="auto",
            density=True,
            label="detected true data width",
            alpha=0.5,
        )
        axes[2].hist(
            detected_pulses_fluence,
            bins="auto",
            density=True,
            label="fake data fluence",
            alpha=0.5,
        )
        plt.legend()
        plt.show()

    print("len detected", len(detected_pulses_snr))
    print("generated", len(total_pulses_snr))
    print("mean", np.mean(total_pulses_snr), "variance", np.std(total_pulses_snr) ** 2)
    print("detection error snr", sigma_snr)
    print("detection error width", sigma_width)
    print("detection error fluence", sigma_fluence)

    out_fn = os.path.join(
        out_fol,
        f"simulated_{mode}_{detected_req}_{mu_ln}_{std_ln}_{w_mu_ln}_{w_std_ln}_{a}.dill",
    )
    yaml_fn = os.path.join(
        out_fol,
        f"simulated_{mode}_{detected_req}_{mu_ln}_{std_ln}_{w_mu_ln}_{w_std_ln}_{a}.yaml",
    )
    if not os.path.exists(out_fol):
        os.makedirs(out_fol)
    # dumps the data into a file
    process_data(
        dill_file,
        detected_pulses_snr,
        detected_pulses_width,
        detected_pulses_fluence,
        out_fn,
    )
    if mode == "Lognorm":
        write_yaml(
            mu_ln,
            std_ln,
            w_mu_ln,
            w_std_ln,
            a,
            len(total_pulses_snr),
            inj_file,
            yaml_fn,
            snr_cutoff = snr_cutoff,
            width_cutoff = width_cutoff,
        )
    elif mode == "Exp":
        write_yaml_exp(mu_ln, a, len(total_pulses_snr), inj_file, yaml_fn)

    if plot:
        snr_array = np.linspace(0, 20, 1000)
        width_array = np.linspace(0, 20, 1001) * 1e-3
        # likelihood, p_det = sb.first_plot(detected_pulses_snr, detected_pulses_width,
        #                                                                                             mu_ln, std_ln,
        #                                                                                             w_mu_ln, w_std_ln,
        #                                                                                             sigma_amp = sigma_snr,
        #                                                                                             sigma_w = sigma_width,
        #                                                                                             a=a,
        #                                                                                             lower_c=lower,
        #                                                                                             upper_c=upper,)
        likelihood, p_det = sb.first_plot(
            snr_array,
            width_array,
            mu_ln,
            std_ln,
            w_mu_ln,
            w_std_ln,
            sigma_amp=sigma_snr,
            sigma_w=sigma_width,
            a=a,
            lower_c=lower,
            upper_c=upper,
        )
        likelihood_norm = likelihood / np.trapz(
            np.trapz(likelihood, snr_array, axis=1), width_array
        )
        # make a 2d histogram of the detections
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        h, xedges, yedges, mesh = axes[0].hist2d(
            detected_pulses_snr, detected_pulses_width, bins=50, density=True
        )
        # colorbar
        cbar = fig.colorbar(mesh, ax=axes[0])
        cbar.ax.set_ylabel("density")
        axes[0].set_xlabel("snr")
        axes[0].set_ylabel("width")
        axes[0].set_title("detected pulses")
        mesh = axes[1].pcolormesh(snr_array, width_array, likelihood_norm)
        # axes[1].scatter(detected_pulses_snr, detected_pulses_width,c=likelihood,s=10)
        cbar = fig.colorbar(mesh, ax=axes[1])
        cbar.ax.set_ylabel("likelihood")
        axes[1].set_xlabel("snr")
        axes[1].set_ylabel("width")
        axes[1].set_title("likelihood")
        # apply axes[0] limits to axes[1]
        axes[1].set_xlim(axes[0].get_xlim())
        axes[1].set_ylim(axes[0].get_ylim())
        plt.show()

        # plot the marginalised distributions
        marg_l_amp = np.trapz(likelihood_norm, width_array, axis=0)
        marg_l_w = np.trapz(likelihood_norm, snr_array, axis=1)
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        n, bins, ax = axes[0].hist(
            detected_pulses_width, bins="auto", density=True, label="detected"
        )
        axes[0].plot(width_array, marg_l_w, label="likelihood")
        axes[0].set_xlabel("width")
        axes[0].set_ylabel("density")
        axes[0].set_title("marginalised over amp")
        axes[0].legend()
        axes[1].hist(detected_pulses_snr, bins="auto", density=True, label="detected")
        axes[1].plot(snr_array, marg_l_amp, label="likelihood")
        axes[1].set_xlabel("snr")
        axes[1].set_ylabel("density")
        axes[1].set_title("marginalised over width")
        axes[1].legend()
        plt.show()


def write_yaml(mu, std, mu_w, std_w, a, N, inj_file, output_fn, snr_cutoff=2,width_cutoff=5e-3):
    mu = float(mu)
    mu_arr = [mu - 1, mu + 1]
    mu_w = float(mu_w)
    mu_w_arr = [mu_w - 1, mu_w + 1]
    data = {
        "logn_N_range": [-1, (N) * 2],
        "logn_mu_range": list(mu_arr),
        "logn_std_range": [std - 0.5, std + 0.5],
        "logn_mu_w_range": list(mu_w_arr),
        "logn_std_w_range": [std_w - 0.5, std_w + 0.5],
        "exp_N_range": [-1, N * 2],
        "exp_k_range": [0.1, 10],
        "detection_curve": inj_file,
        "snr_thresh": snr_cutoff,
        "width_thresh": width_cutoff,
        "a": a,
        "N": N,
        "mu": mu,
        "mu_w": mu_w,
        "std": std,
        "std_w": std_w,
    }
    with open(output_fn, "w") as my_file:
        yaml.dump(data, my_file)


def write_yaml_exp(k, a, N, inj_file, output_fn, snr_thresh=2):
    # in this case mu is the k variable for the exponential distribution
    k = float(k)
    k_low = k - 1
    if k_low < 0:
        k_low = 0.1
    k_arr = [k_low, k + 1]

    data = {
        "exp_N_range": [-1, N * 2],
        "exp_k_range": list(k_arr),
        "logn_N_range": [-1, N * 2],
        "logn_mu_range": [-2, 2],
        "logn_std_range": [0.01, 2],
        "detection_curve": inj_file,
        "snr_thresh": snr_thresh,
        "a": a,
        "N": N,
        "k": k,
    }
    with open(output_fn, "w") as my_file:
        yaml.dump(data, my_file)


def process_data(dill_file, detected_snr, detected_width, detected_fluence, output_fn):
    """
    Process the data from a Dill file and create a fake data file.

    Args:
        dill_file (str): Path to the Dill file containing the data.
        plot (bool, optional): Whether to generate a histogram plot. Defaults to True.

    Returns:
        None
    """

    with open(dill_file, "rb") as inf:
        det_class = dill.load(inf)
    filfiles = np.full(len(detected_snr), "abc", dtype=str)
    maskfn = np.full(len(detected_snr), "abc", dtype=str)
    dms = np.full(len(detected_snr), 123, dtype=float)
    toas = np.full(len(detected_snr), 123, dtype=float)

    inject_obj_arr = []
    for snr, width, fluence in zip(detected_snr, detected_width, detected_fluence):
        # create a new inject_obj and just fill with fake data
        temp = inject_obj()
        temp.det_snr = snr
        temp.det_fluence = fluence
        temp.det_std = width

        temp.det_amp = snr
        temp.fluence_amp = snr
        temp.noise_std = snr
        inject_obj_arr.append(temp)
    inj_obj_arr = np.array(inject_obj_arr)

    det_class.filfiles = filfiles
    det_class.mask_fn = maskfn
    det_class.dms = dms
    det_class.toas = toas
    det_class.sorted_pulses = inject_obj_arr

    with open(output_fn, "wb") as of:
        dill.dump(det_class, of)


parser = argparse.ArgumentParser(description="Simulate some pulses")
parser.add_argument("-a", type=float, default=0, help="offset value")
parser.add_argument("-d", type=str, default="", help="dill_file")
parser.add_argument("-inj_file", type=str, help="injection_file.dill")
parser.add_argument("-s", type=float, default=0.5, help="std value")
parser.add_argument(
    "-mode",
    type=str,
    default="Lognorm",
    help="Choose the distribution you want the fake data to have, the options are Lognorm, Gauss, Exp",
)
parser.add_argument("-of", type=str, default="simulated_dir", required=True)
parser.add_argument("-detected_req", type=int, default=10000)
args = parser.parse_args()
a = args.a
dill_file = args.d
inj_file = args.inj_file
output_fol = args.of
mode = args.mode
# load the detection file
# sb = statistics_basic.statistics_basic(inj_file,plot=True)
if mode == "Lognorm":
    mu_arr = np.linspace(-0.5, 1, 10)
    std_arr = [args.s]
    mu_w_arr = np.linspace(-6.5, -4.5, 10)
    std_w_arr = [0.3]
elif mode == "Exp":
    k_arr = np.linspace(0.5, 2, 5)
    # use mu as k_arr
    mu_arr = k_arr
    # the std arr is ignored for exp distribution
    std_arr = [0.5]

detected_req = args.detected_req
print("mu", mu_arr, "std", std_arr, "a", a, "detected_req", detected_req)
from inject_stats import inject_obj
from numpy.random import normal
from statistics import statistics_ln

# from statistics import lognorm_dist
# from statistics import mean_var_to_mu_std
# from statistics import mu_std_to_mean_var
# from statistics_exp import k_to_mean_var
#
if __name__ == "__main__":
    # simulate pulses one at a time
    snr_cutoff = 2.0
    width_cutoff = 2e-3
    sb = statistics_ln(inj_file, plot=True, snr_cutoff=snr_cutoff, width_cutoff=width_cutoff)
    sb.convolve_p_detect()
    for mu in mu_arr:
        for w_mu_ln in mu_w_arr:
            std = std_arr[0]
            w_std_ln = std_w_arr[0]
            lower = 0
            if mode == "Lognorm":
                # use the upper cutoff to be 50x the mean
                mean, var = sb.mu_std_to_mean_var(mu, std)
                median = np.exp(mu)
                # set upper to 50x median
                upper = 50 * median
            elif mode == "Exp":
                mean, var = k_to_mean_var(mu)
                median = np.log(2) / mu
                upper = 50 * median
            simulate_and_process_data(
                detected_req,
                mode,
                mu,
                std,
                w_mu_ln,
                w_std_ln,
                lower,
                upper,
                sb,
                a,
                inj_file,
                dill_file,
                # plot=True,
                plot=False,
                out_fol=output_fol,
                snr_cutoff=snr_cutoff,
                width_cutoff=width_cutoff,
            )
