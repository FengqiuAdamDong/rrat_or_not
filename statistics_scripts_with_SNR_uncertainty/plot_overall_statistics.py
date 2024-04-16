import numpy as np
import matplotlib.pyplot as plt
import argparse
import smplotlib
from bayes_factor_LNLN import read_config
import pygedm
import csv


def get_best_fit_values(
    dynesty_results,
    calibrated_samples,
    calibrated_samples_lower,
    calibrated_samples_upper,
):
    from dynesty.utils import quantile
    from dynesty.utils import mean_and_cov

    if calibrated_samples is not None:
        samples = calibrated_samples
        samples_lower = calibrated_samples_lower
        samples_upper = calibrated_samples_upper
        print("Using calibrated samples")
    else:
        samples = dynesty_results.samples
        samples_lower = dynesty_results.samples
        samples_upper = dynesty_results.samples
    importance_weights = dynesty_results.importance_weights()
    quantiles = []
    quantiles_lower = []
    quantiles_upper = []
    lower_lim = 0.16
    upper_lim = 0.84
    for i in range(samples.shape[1]):
        quantiles.append(
            quantile(
                samples[:, i], [lower_lim, 0.5, upper_lim], weights=importance_weights
            )
        )
        quantiles_lower.append(
            quantile(
                samples_lower[:, i],
                [lower_lim, 0.5, upper_lim],
                weights=importance_weights,
            )
        )
        quantiles_upper.append(
            quantile(
                samples_upper[:, i],
                [lower_lim, 0.5, upper_lim],
                weights=importance_weights,
            )
        )
    quantiles = np.array(quantiles)
    mean, cov = mean_and_cov(samples, weights=importance_weights)
    return quantiles, quantiles_lower, quantiles_upper, mean, cov


def load_data(fn):
    data = np.load(fn, allow_pickle=True)
    if "calibrated_samples" in data.keys():
        calibrated_samples = data["calibrated_samples"]
        calibrated_samples_lower = data["calibrated_samples_lower"]
        calibrated_samples_upper = data["calibrated_samples_upper"]
    else:
        calibrated_samples = None
        calibrated_samples_lower = None
        calibrated_samples_upper = None
    data = data["results"].tolist()
    return data, calibrated_samples, calibrated_samples_lower, calibrated_samples_upper


def convert_dm_to_distance(dm_arr, ra_arr, dec_arr):
    import astropy
    from astropy.coordinates import SkyCoord
    from astropy import units as u

    distance_arr = []
    for dm, ra, dec in zip(dm_arr, ra_arr, dec_arr):
        c = SkyCoord(ra=ra * u.degree, dec=dec * u.degree)
        # get the ra and dec in galactic
        gl = c.galactic.l.degree
        gb = c.galactic.b.degree
        distance, tau = pygedm.dm_to_dist(gl, gb, dm)
        # convert distance to kpc
        distance = distance.to(u.kpc)
        distance_arr.append(distance)
    return distance_arr


def write_table(quantiles, quantiles_lower, quantiles_upper, names):
    with open("fit_results_table.csv", mode="w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Pulsar Name",
                "$\mu_S$",
                "$\sigma_S$",
                "$\mu_W$",
                "$\sigma_W$",
                "N",
            ]
        )
        writer.writerow(
            [
                "",
                "ln(Jy)",
                "",
                "lm(s)",
                "",
                ""
            ]
        )
        for q, q_lower, q_upper, name in zip(
            quantiles, quantiles_lower, quantiles_upper, names
        ):
            mu_snr = q[0][1]
            mu_snr_stats_lower_err = q_lower[0][0] - mu_snr
            mu_snr_stats_upper_err = q_upper[0][2] - mu_snr

            std_snr = q[1][1]
            std_snr_stats_lower_err = q_lower[1][0] - std_snr
            std_snr_stats_upper_err = q_upper[1][2] - std_snr

            mu_w = q[2][1]
            mu_w_stats_lower_err = q_lower[2][0] - mu_w
            mu_w_stats_upper_err = q_upper[2][2] - mu_w

            std_w = q[3][1]
            std_w_stats_lower_err = q_lower[3][0] - std_w
            std_w_stats_upper_err = q_upper[3][2] - std_w

            N = int(q[4][1])
            N_stats_lower_err = int(q_lower[4][0]) - N
            N_stats_upper_err = int(q_upper[4][2]) - N

            mu_snr_flux_lower = q_lower[0][1]
            mu_snr_flux_upper = q_upper[0][1]

            mu_snr_flux_lower_err = mu_snr_flux_lower - mu_snr
            mu_snr_flux_upper_err = mu_snr_flux_upper - mu_snr

            mu_snr_str = (
                "$"
                + str(round(mu_snr, 3))
                + "^{+"
                + str(round(mu_snr_stats_upper_err, 3))
                + "+"
                + str(round(mu_snr_flux_upper_err, 3))
                + "}"
                + "_{-"
                + str(round(mu_snr_stats_lower_err, 3))
                + "-"
                + str(round(mu_snr_flux_lower_err, 3))
                + "}"
                + "$"
            )
            std_snr_str = (
                "$"
                + str(round(std_snr, 3))
                + "^{+"
                + str(round(std_snr_stats_upper_err, 3))
                + "}"
                + "_{-"
                + str(round(std_snr_stats_lower_err, 3))
                + "}$"
            )
            mu_w_str = (
                "$"
                + str(round(mu_w, 3))
                + "^{+"
                + str(round(mu_w_stats_upper_err, 3))
                + "}"
                + "_{-"
                + str(round(mu_w_stats_lower_err, 3))
                + "}$"
            )
            std_w_str = (
                "$"
                + str(round(std_w, 3))
                + "^{+"
                + str(round(std_w_stats_upper_err, 3))
                + "}"
                + "_{-"
                + str(round(std_w_stats_lower_err, 3))
                + "}$"
            )
            N_str = (
                "$"
                + str(N)
                + "^{+"
                + str(N_stats_upper_err)
                + "}"
                + "_{-"
                + str(N_stats_lower_err)
                + "}$"
            )

            writer.writerow(
                [
                    name,
                    mu_snr_str,
                    std_snr_str,
                    mu_w_str,
                    std_w_str,
                    N_str,
                ]
            )


def process_npz(npz_files, yaml_files, pulsar_names, dm, ra, dec):
    means = []
    covs = []
    quantiles = []
    quantiles_lower = []
    quantiles_upper = []
    N_cap = []
    associated_DM = []
    associated_RA = []
    associated_DEC = []
    pulsar_names_list = []
    for npz_file, yaml_file in zip(npz_files, yaml_files):
        print(npz_file, yaml_file)

        (
            data,
            calibrated_samples,
            calibrated_samples_lower,
            calibrated_samples_upper,
        ) = load_data(npz_file)
        quantile, quantile_lower, quantile_upper, mean, cov = get_best_fit_values(
            data, calibrated_samples, calibrated_samples_lower, calibrated_samples_upper
        )
        print(mean)
        means.append(mean)
        covs.append(np.diag(cov))
        quantiles.append(quantile)
        quantiles_lower.append(quantile_lower)
        quantiles_upper.append(quantile_upper)
        _, logn_N_range, _, _, _, _, _, _, _, _, _ = read_config(yaml_file)
        N_cap.append(logn_N_range[1])
        pulsar_found = False
        for i, name in enumerate(pulsar_names):
            if name in npz_file:
                associated_DM.append(dm[i])
                associated_RA.append(ra[i])
                associated_DEC.append(dec[i])
                pulsar_found = True
                pulsar_names_list.append(name)
                break
        if not pulsar_found:
            print(f"Could not find pulsar {npz_file} in the list of pulsar names")
    quantiles = np.array(quantiles)
    N_cap = np.array(N_cap)
    associated_DM = np.array(associated_DM)
    associated_RA = np.array(associated_RA)
    associated_DEC = np.array(associated_DEC)
    # for each pulsar find the DM distance
    associated_distance = convert_dm_to_distance(
        associated_DM, associated_RA, associated_DEC
    )

    # means = np.array(means)
    # covs = np.array(covs)
    # mu_snr = [mean[0] for mean in means]
    # mu_snr_err = [cov[0] for cov in covs]
    # std_snr = [mean[1] for mean in means]
    # std_snr_err = [cov[1] for cov in covs]
    # mu_w = [mean[2] for mean in means]
    # mu_w_err = [cov[2] for cov in covs]
    # std_w = [mean[3] for mean in means]
    # std_w_err = [cov[3] for cov in covs]
    # mu_snr = np.array(mu_snr)
    # mu_snr_err = np.array(mu_snr_err)
    # std_snr = np.array(std_snr)
    # std_snr_err = np.array(std_snr_err)
    # mu_w = np.array(mu_w)
    # mu_w_err = np.array(mu_w_err)
    # std_w = np.array(std_w)
    # std_w_err = np.array(std_w_err)

    mu_snr_quantiles = [quantile[0] for quantile in quantiles]
    std_snr_quantiles = [quantile[1] for quantile in quantiles]
    mu_w_quantiles = [quantile[2] for quantile in quantiles]
    std_w_quantiles = [quantile[3] for quantile in quantiles]

    mu_snr_quantiles_lower = [quantile[0] for quantile in quantiles_lower]
    mu_snr_quantiles_upper = [quantile[0] for quantile in quantiles_upper]
    write_table(quantiles, quantiles_lower, quantiles_upper, pulsar_names_list)
    N = [quantile[4] for quantile in quantiles]

    N = np.array(N)
    mu_snr_quantiles = np.array(mu_snr_quantiles)
    std_snr_quantiles = np.array(std_snr_quantiles)
    mu_w_quantiles = np.array(mu_w_quantiles)
    std_w_quantiles = np.array(std_w_quantiles)

    mu_snr_q_val = mu_snr_quantiles[:, 1]
    std_snr_q_val = std_snr_quantiles[:, 1]
    mu_w_q_val = mu_w_quantiles[:, 1]
    std_w_q_val = std_w_quantiles[:, 1]

    mu_snr_q_low = np.abs(mu_snr_quantiles[:, 0] - mu_snr_q_val)
    std_snr_q_low = np.abs(std_snr_quantiles[:, 0] - std_snr_q_val)
    mu_w_q_low = np.abs(mu_w_quantiles[:, 0] - mu_w_q_val)
    std_w_q_low = np.abs(std_w_quantiles[:, 0] - std_w_q_val)

    mu_snr_q_high = np.abs(mu_snr_quantiles[:, 2] - mu_snr_q_val)
    std_snr_q_high = np.abs(std_snr_quantiles[:, 2] - std_snr_q_val)
    mu_w_q_high = np.abs(mu_w_quantiles[:, 2] - mu_w_q_val)
    std_w_q_high = np.abs(std_w_quantiles[:, 2] - std_w_q_val)

    luminosity_distance = []
    for mu_s, d in zip(mu_snr_quantiles, associated_distance):
        luminosity_distance.append(np.array(mu_s) + np.log(d.value**2))  # Jy pc^2
    luminosity_distance = np.array(luminosity_distance)
    mu_lum_dist_q_val = luminosity_distance[:, 1]
    mu_lum_dist_q_low = np.abs(luminosity_distance[:, 0] - mu_lum_dist_q_val)
    mu_lum_dist_q_high = np.abs(luminosity_distance[:, 2] - mu_lum_dist_q_val)

    # plot the 2d histogram of the results
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax[0, 0].hist(mu_snr_q_val, bins="auto")
    ax[0, 0].set_xlabel(r"$\mu_S$")
    ax[0, 1].hist(std_snr_q_val, bins="auto")
    ax[0, 1].set_xlabel(r"$\sigma_S$")
    ax[1, 0].hist(mu_w_q_val, bins="auto")
    ax[1, 0].set_xlabel(r"$\mu_w$ (ln(s))")
    ax[1, 1].hist(std_w_q_val, bins="auto")
    ax[1, 1].set_xlabel(r"$\sigma_w$")
    plt.tight_layout()
    plt.savefig("hists.png")
    null_all = 1 - (N / N_cap[:, np.newaxis])
    # import pdb; pdb.set_trace()
    null_error = np.array(
        [(np.abs(n[2] - n[1]), np.abs(n[0] - n[1])) for n in null_all]
    ).T
    null = np.array([n[1] for n in null_all])

    # fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    # h = ax[0].hist2d(null, mu_snr_q_val, bins=5)
    # # color bar
    # cbar = plt.colorbar(h[3], ax=ax[0])
    # ax[0].set_xlabel("nulling fraction")
    # ax[0].set_ylabel("mu_snr")
    # h = ax[1].hist2d(null, mu_w_q_val, bins=5)
    # cbar = plt.colorbar(h[3], ax=ax[1])
    # ax[1].set_xlabel("nulling fraction")
    # ax[1].set_ylabel("mu_w")
    # plt.tight_layout()
    # plt.savefig("hists2N.png")

    plt.figure()
    plt.hist(null, bins="auto")
    plt.savefig("nulling_fraction.png")
    plt.xlabel("Nulling Fraction")

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    # fig.suptitle("Quantiles")
    ax[0, 0].errorbar(
        null, mu_snr_q_val, yerr=[mu_snr_q_low, mu_snr_q_high], xerr=null_error, fmt="o"
    )
    ax[0, 0].set_xlabel("Nulling Fraction")
    ax[0, 0].set_ylabel(r"$\mu_{\rm S} (ln(Jy))$")
    ax[0, 0].set_xlim(0, 1)
    ax[0, 1].errorbar(
        null, mu_w_q_val, yerr=[mu_w_q_low, mu_w_q_high], xerr=null_error, fmt="o"
    )
    ax[0, 1].set_xlabel("Nulling Fraction")
    ax[0, 1].set_ylabel(r"$\mu_{\rm W} (ln(s))$")
    ax[0, 1].set_xlim(0, 1)
    ax[1, 0].errorbar(
        null,
        std_snr_q_val,
        yerr=[std_snr_q_low, std_snr_q_high],
        xerr=null_error,
        fmt="o",
    )
    ax[1, 0].set_xlabel("Nulling Fraction")
    ax[1, 0].set_ylabel(r"$\sigma_{\rm S}$")
    ax[1, 0].set_xlim(0, 1)
    ax[1, 1].errorbar(
        null, std_w_q_val, yerr=[std_w_q_low, std_w_q_high], xerr=null_error, fmt="o"
    )
    ax[1, 1].set_xlabel("Nulling Fraction")
    ax[1, 1].set_ylabel(r"$\sigma_{\rm W}$")
    ax[1, 1].set_xlim(0, 1)
    plt.tight_layout()
    plt.savefig("everythingvsnulling.png")

    # plot mu snr vs std snr
    fig, ax = plt.subplots(2, 3, figsize=(10, 10))
    ax[0, 0].errorbar(
        mu_snr_q_val,
        std_snr_q_val,
        yerr=[std_snr_q_low, std_snr_q_high],
        xerr=[mu_snr_q_low, mu_snr_q_high],
        fmt="o",
    )
    ax[0, 0].set_xlabel(r"$\mu_{\rm S} (ln(Jy))$")
    ax[0, 0].set_ylabel(r"$\sigma_{\rm S}$")
    ax[0, 1].errorbar(
        mu_w_q_val,
        std_w_q_val,
        yerr=[std_w_q_low, std_w_q_high],
        xerr=[mu_w_q_low, mu_w_q_high],
        fmt="o",
    )
    ax[0, 1].set_xlabel(r"$\mu_{\rm W} (ln(s))$")
    ax[0, 1].set_ylabel(r"$\sigma_{\rm W}$")
    ax[1, 0].errorbar(
        mu_snr_q_val,
        std_w_q_val,
        yerr=[std_w_q_low, std_w_q_high],
        xerr=[mu_snr_q_low, mu_snr_q_high],
        fmt="o",
    )
    ax[1, 0].set_xlabel(r"$\mu_{\rm S} (ln(Jy))$")
    ax[1, 0].set_ylabel(r"$\sigma_{\rm W}$")
    ax[1, 1].errorbar(
        mu_w_q_val,
        std_snr_q_val,
        yerr=[std_snr_q_low, std_snr_q_high],
        xerr=[mu_w_q_low, mu_w_q_high],
        fmt="o",
    )
    ax[1, 1].set_xlabel(r"$\mu_{\rm W} (ln(s))$")
    ax[1, 1].set_ylabel(r"$\sigma_{\rm S}$")
    ax[0, 2].errorbar(
        mu_snr_q_val,
        mu_w_q_val,
        yerr=[mu_w_q_low, mu_w_q_high],
        xerr=[mu_snr_q_low, mu_snr_q_high],
        fmt="o",
    )
    ax[0, 2].set_xlabel(r"$\mu_{\rm S} (ln(Jy))$")
    ax[0, 2].set_ylabel(r"$\mu_{\rm W} (ln(s))$")
    ax[1, 2].errorbar(
        std_snr_q_val,
        std_w_q_val,
        yerr=[std_w_q_low, std_w_q_high],
        xerr=[std_snr_q_low, std_snr_q_high],
        fmt="o",
    )
    ax[1, 2].set_xlabel(r"$\sigma_{\rm S}$")
    ax[1, 2].set_ylabel(r"$\sigma_{\rm W}$")

    plt.tight_layout()
    plt.savefig("mu_vs_std.png")

    # plot the luminosity distance against everything
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax[0, 0].errorbar(
        mu_lum_dist_q_val,
        std_snr_q_val,
        yerr=[std_snr_q_low, std_snr_q_high],
        xerr=[mu_lum_dist_q_low, mu_lum_dist_q_high],
        fmt="o",
    )
    ax[0, 0].set_xlabel(r"$\mu_{\rm L} (ln(Jy kpc^2))$")
    ax[0, 0].set_ylabel(r"$\sigma_{\rm L}$")
    ax[0, 1].errorbar(
        mu_lum_dist_q_val,
        std_w_q_val,
        yerr=[std_w_q_low, std_w_q_high],
        xerr=[mu_lum_dist_q_low, mu_lum_dist_q_high],
        fmt="o",
    )
    ax[0, 1].set_xlabel(r"$\mu_{\rm L} (ln(Jy kpc^2))$")
    ax[0, 1].set_ylabel(r"$\sigma_{\rm W}$")
    ax[1, 0].errorbar(
        mu_lum_dist_q_val,
        mu_w_q_val,
        yerr=[mu_w_q_low, mu_w_q_high],
        xerr=[mu_lum_dist_q_low, mu_lum_dist_q_high],
        fmt="o",
    )
    ax[1, 0].set_xlabel(r"$\mu_{\rm L} (ln(Jy kpc^2))$")
    ax[1, 0].set_ylabel(r"$\mu_{\rm W} (ln(s))$")

    plt.show()


def read_pulsar_pop(filename):
    # read the pulsar population filename
    with open(filename, "r") as f:
        reader = csv.reader(f)
        pulsar_name = []
        rajd = []
        decjd = []
        dm = []
        for i, row in enumerate(reader):
            if i == 0:
                continue
            pulsar_name.append(row[6])
            rajd.append(float(row[8]))
            decjd.append(float(row[9]))
            dm.append(float(row[11]))
    pulsar_name = np.array(pulsar_name)
    rajd = np.array(rajd)
    decjd = np.array(decjd)
    dm = np.array(dm)
    return pulsar_name, rajd, decjd, dm


if __name__ == "__main__":
    import sys

    npzs = sys.argv[1:]
    npz_base = [name.split(".")[0] for name in npzs]
    yamls = [name + ".yaml" for name in npz_base]
    pulsar_table = "pulsar_pop_sheet.csv"
    pulsar_name, rajd, decjd, dm = read_pulsar_pop(pulsar_table)
    process_npz(npzs, yamls, pulsar_name, dm, rajd, decjd)
