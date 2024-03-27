#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import dynesty
import sys
from bayes_factor_NS_LN_no_a_single import loglikelihood
from bayes_factor_NS_LN_no_a_single import pt_Uniform_N
from scipy.interpolate import RegularGridInterpolator
import yaml
import smplotlib
import os


def Ntonull(N):
    """
    Convert the number of events to the nulling fraction
    """
    nulling_frac = 1 - (N / (obs_time / period))
    # print("nulling frac",nulling_frac,"N",N)
    return nulling_frac


def nulltoN(null):
    N = (1 - null) * (obs_time / period)
    # print(obs_time,period,"obs time, period")
    # print("N",N,"nulling frac",null)
    return N


def smoothTriangle(data, degree):
    triangle = np.concatenate(
        (np.arange(degree + 1), np.arange(degree)[::-1])
    )  # up then down
    smoothed = []

    for i in range(degree, len(data) - degree * 2):
        point = data[i : i + len(triangle)] * triangle
        smoothed.append(np.sum(point) / np.sum(triangle))
    # Handle boundaries
    smoothed = [smoothed[0]] * int(degree + degree / 2) + smoothed
    while len(smoothed) < len(data):
        smoothed.append(smoothed[-1])
    return smoothed


class dynesty_plot:
    """
    Class to plot dynesty results"""

    def __init__(self, filenames):
        self.filename = filenames
        # get the parameters from the first filename
        split = filenames[0].split("_")
        self.dist = split[1]
        # make self.dist lower case
        self.dist = self.dist.lower()
        try:
            self.dets = int(split[2])
        except:
            self.dets = 2

    def load_filenames(self):
        """
        Load the filenames into a list of data objects
        """
        self.data = []
        delete_fn = []
        for f in self.filename:
            # check that the png was made i.e. the run finished
            # png_fn = f.replace("_logn.h5",".dill")+"_logn_a_corner.png"
            # if not os.path.isfile(png_fn):
            #     print(f"WARNING: {png_fn} does not exist, skipping")
            #     #delete f from self.filename
            #     delete_fn.append(f)
            #     continue

            if f.endswith(".npz"):
                self.data.append(np.load(f, allow_pickle=True)["results"].tolist())
                print(f)
                evidence = self.data[-1].logz[-1]
                evidence_error = self.data[-1].logzerr[-1]
                print(f"evidence for {evidence}+-{evidence_error}")
            elif f.endswith(".h5"):
                self.data.append(dynesty.NestedSampler.restore(f).results)
                evidence = self.data[-1].logz[-1]
                evidence_error = self.data[-1].logzerr[-1]
                print(f"evidence for {evidence}+-{evidence_error}")
        for f in delete_fn:
            self.filename.remove(f)

    def plot_bayes_ratio_logn(self):
        mu_arr = []
        evidence_diff_arr = []
        evidence_diff_err = []
        for fn, logn_sampler in zip(self.filename, self.data):
            # get the logn version of the filename
            split = fn.split("_")
            exp_fn = "_".join(split[:-1]) + "_exp.h5"
            exp_sampler = dynesty.NestedSampler.restore(exp_fn)
            # get the logn evidence
            exp_evidence = exp_sampler.logz[-1]
            exp_evidence_error = exp_sampler.logzerr[-1]

            logn_evidence = logn_sampler.logz[-1]
            logn_evidence_error = logn_sampler.logzerr[-1]
            # propagate the errors
            evidence_error = np.sqrt(exp_evidence_error**2 + logn_evidence_error**2)

            print(
                "exp evidence",
                exp_evidence,
                "logn evidence",
                logn_evidence,
                "comparison",
                logn_evidence - exp_evidence,
                "error",
                evidence_error,
            )
            # get the k value from fn
            mu = float(fn.split("_")[3])
            mu_arr.append(mu)
            evidence_diff_arr.append(logn_evidence - exp_evidence)
        plt.figure()
        plt.errorbar(mu_arr, evidence_diff_arr, yerr=evidence_error, linestyle="none")
        plt.xlabel(r"$mu$")
        plt.ylabel(r"$\ln(Z_{logn})-\ln(Z_{exp})$")
        plt.savefig(f"{self.dets}_evidence_diff_logn.pdf")

    def plot_bayes_ratio_exp(self):
        k_arr = []
        evidence_diff_arr = []

        for fn, exp_sampler in zip(self.filename, self.data):
            # get the logn version of the filename
            split = fn.split("_")
            logn_fn = "_".join(split[:-1]) + "_logn.h5"
            logn_sampler = dynesty.NestedSampler.restore(logn_fn)
            # get the logn evidence
            exp_evidence = exp_sampler.logz[-1]
            exp_evidence_error = exp_sampler.logzerr[-1]

            logn_evidence = logn_sampler.logz[-1]
            logn_evidence_error = logn_sampler.logzerr[-1]

            evidence_error = np.sqrt(exp_evidence_error**2 + logn_evidence_error**2)
            print(
                "exp evidence",
                exp_evidence,
                "logn evidence",
                logn_evidence,
                "comparison",
                exp_evidence - logn_evidence,
                "error",
                evidence_error,
            )
            # get the k value from fn
            k = float(fn.split("_")[3])
            k_arr.append(k)
            evidence_diff_arr.append(exp_evidence - logn_evidence)
        plt.figure()
        plt.errorbar(k_arr, evidence_diff_arr, yerr=evidence_error, linestyle="none")
        plt.xlabel(r"$k$")
        plt.ylabel(r"$\ln(Z_{exp})-\ln(Z_{logn})$")
        plt.savefig(f"{self.dets}_evidence_diff_exp.pdf")

    def plot_corner(self, labels=None, plot=False):
        """
        Plot the corner plot of the samples
        """
        from dynesty import plotting as dyplot

        means = []
        stds = []
        quantiles = []
        for f, d in zip(self.filename, self.data):
            if plot:
                corner = dyplot.cornerplot(d, labels=labels)
                if (period > 0) & (obs_time > 0):
                    # if these two global variables are set
                    try:
                        N_ax = corner[1][2][2]
                        N_sax = N_ax.secondary_xaxis(
                            "top", functions=(Ntonull, nulltoN)
                        )
                        N_sax.set_xlabel(r"Nulling fraction")
                    except:
                        N_ax = corner[1][1][1]
                        N_sax = N_ax.secondary_xaxis(
                            "top", functions=(Ntonull, nulltoN)
                        )
                        N_sax.set_xlabel(r"Nulling fraction")
                plt.savefig(f.strip(".h5") + "_corner.pdf")

                plt.savefig(f.strip(".h5") + "_corner.png")
                plt.close()
            mean, cov = dynesty.utils.mean_and_cov(d.samples, d.importance_weights())
            quantile = []
            for i in range(d.samples.shape[1]):
                quantile.append(
                    np.array(
                        dynesty.utils.quantile(
                            d.samples[:, i],
                            [0.159, 0.50, 0.841],
                            d.importance_weights(),
                        )
                    )
                )
            quantile = np.array(quantile)
            diag_cov = np.diag(cov)
            std = np.sqrt(diag_cov)
            means.append(mean)
            stds.append(std)
            quantile = np.array(quantile)
            quantiles.append(quantile)
            # print(means)
            # print(std)
        self.means = np.array(means)
        self.stds = np.array(stds)
        self.quantiles = np.array(quantiles)

    def plot_accuracy_logn(self):
        """
        Plot the accuracy of the results
        """
        true_centres = []

        for fn, centre, errors in zip(self.filename, self.means, self.stds):
            # get the mu and sigma from the filename
            split = fn.split("_")
            # join all but the last element
            yaml_file = "_".join(split[:-1]) + ".yaml"
            # load the yaml file
            with open(yaml_file) as f:
                yaml_data = yaml.safe_load(f)
                true_centres.append(
                    np.array(
                        [
                            yaml_data["mu"],
                            yaml_data["std"],
                            yaml_data["mu_w"],
                            yaml_data["std_w"],
                            yaml_data["N"],
                        ]
                    )
                )

        # plot the first element of the ratios
        true_mus = [r[0] for r in true_centres]
        true_sigmas = [r[1] for r in true_centres]
        true_mus_w = [r[2] for r in true_centres]
        true_sigmas_w = [r[3] for r in true_centres]
        true_Ns = [r[4] for r in true_centres]

        mus = np.zeros(len(true_mus))
        mu_errs = np.zeros((2, len(true_mus)))
        sigmas = np.zeros(len(true_mus))
        sigma_errs = np.zeros((2, len(true_mus)))
        mus_w = np.zeros(len(true_mus))
        mu_errs_w = np.zeros((2, len(true_mus)))
        sigmas_w = np.zeros(len(true_mus))
        sigma_errs_w = np.zeros((2, len(true_mus)))
        Ns = np.zeros(len(true_mus))
        N_errs = np.zeros((2, len(true_mus)))
        for i, quantile in enumerate(self.quantiles):
            mus[i] = quantile[0][1]
            mu_errs[:, i] = np.array(
                [quantile[0][1] - quantile[0][0], quantile[0][2] - quantile[0][1]]
            )
            sigmas[i] = quantile[1][1]
            sigma_errs[:, i] = np.array(
                [quantile[1][1] - quantile[1][0], quantile[1][2] - quantile[1][1]]
            )
            mus_w[i] = quantile[2][1]
            mu_errs_w[:, i] = np.array(
                [quantile[2][1] - quantile[2][0], quantile[2][2] - quantile[2][1]]
            )
            sigmas_w[i] = quantile[3][1]
            sigma_errs_w[:, i] = np.array(
                [quantile[3][1] - quantile[3][0], quantile[3][2] - quantile[3][1]]
            )
            Ns[i] = quantile[4][1]
            N_errs[:, i] = np.array(
                [quantile[4][1] - quantile[4][0], quantile[4][2] - quantile[4][1]]
            )

        plt.figure()
        max_mu = max([max(true_mus), max(mus)])
        min_mu = min([min(true_mus), min(mus)])
        plt.errorbar(
            true_mus, mus, yerr=mu_errs, label="mu", linestyle="None", marker="o"
        )
        x = np.linspace(min_mu, max_mu, 100)
        plt.plot(x, x, "r--")
        plt.xlabel(r"True $\mu$")
        plt.ylabel(r"recovered $\mu$")
        plt.title(f"{self.dets} detections true sigma = {true_sigmas[0]}")
        plt.savefig(f"{self.dets}_mus.pdf")
        plt.savefig(f"{self.dets}_mus.png")
        plt.figure()
        true_sigmas = np.array(true_sigmas)
        sigmas = np.array(sigmas)
        plt.errorbar(
            range(len(sigmas)),
            sigmas / true_sigmas,
            yerr=sigma_errs / true_sigmas,
            label="sigma",
            linestyle="None",
            marker="o",
        )
        max_sigma = max([max(true_sigmas), max(sigmas)])
        # x = np.linspace(0,max_sigma,100)
        # plt.plot(x,x,'r--')
        plt.xlabel(r"index")
        plt.ylabel(r"recovered $\sigma$/True $\sigma$")
        plt.title(f"{self.dets} detections true sigma = {true_sigmas[0]}")
        plt.savefig(f"{self.dets}_sigmas.pdf")
        plt.savefig(f"{self.dets}_sigmas.png")
        plt.figure()
        max_mu_w = max([max(true_mus_w), max(mus_w)])
        min_mu_w = min([min(true_mus_w), min(mus_w)])
        plt.errorbar(
            true_mus_w,
            mus_w,
            yerr=mu_errs_w,
            label="mu_w",
            linestyle="None",
            marker="o",
        )
        x = np.linspace(min_mu_w, max_mu_w, 100)
        plt.plot(x, x, "r--")
        plt.xlabel(r"True $\mu_w$")
        plt.ylabel(r"recovered $\mu_w$")
        plt.title(f"{self.dets} detections true sigma = {true_sigmas_w[0]}")
        plt.savefig(f"{self.dets}_mus_w.pdf")
        plt.savefig(f"{self.dets}_mus_w.png")

        plt.figure()
        true_sigmas_w = np.array(true_sigmas_w)
        sigmas_w = np.array(sigmas_w)
        plt.errorbar(
            range(len(sigmas_w)),
            sigmas_w / true_sigmas_w,
            yerr=sigma_errs_w / true_sigmas_w,
            label="sigma_w",
            linestyle="None",
            marker="o",
        )
        max_sigma_w = max([max(true_sigmas_w), max(sigmas_w)])
        plt.xlabel(r"index")
        plt.ylabel(r"recovered $\sigma_w$/True $\sigma_w$")
        plt.title(f"{self.dets} detections true sigma_w = {true_sigmas_w[0]}")
        plt.savefig(f"{self.dets}_sigmas_w.pdf")
        plt.savefig(f"{self.dets}_sigmas_w.png")

        plt.figure()
        max_N = max([max(true_Ns), max(Ns)])
        x = np.linspace(0, max_N, 100)
        plt.errorbar(true_Ns, Ns, yerr=N_errs, label="N", linestyle="None", marker="o")
        # loglog plot
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("True N")
        plt.ylabel("recovered N")
        plt.title(f"{self.dets} detections true sigma = {true_sigmas[0]}")
        plt.plot(x, x, "r--")
        plt.savefig(f"{self.dets}_logn_Ns.pdf")
        plt.savefig(f"{self.dets}_logn_Ns.png")
        # plt.scatter(true_Ns,N_ratios,label="N")
        plt.legend()

    def plot_accuracy_exp(self):
        """
        Plot the accuracy of the results
        """
        true_centres = []

        for fn, centre, errors in zip(self.filename, self.means, self.stds):
            # get the mu and sigma from the filename
            split = fn.split(".dill")
            # join all but the last element
            yaml_file = split[0] + ".yaml"
            # load the yaml file
            with open(yaml_file) as f:
                yaml_data = yaml.safe_load(f)
                true_centres.append(
                    np.array([yaml_data["k"], yaml_data["k_w"], yaml_data["N"]])
                )
        # plot the first element of the ratios
        true_ks = [r[0] for r in true_centres]
        ks = [r[0] for r in self.means]
        k_errs = [r[0] for r in self.stds]

        true_k2s = [r[1] for r in true_centres]
        k2s = [r[1] for r in self.means]
        k2_errs = [r[1] for r in self.stds]

        true_Ns = [r[2] for r in true_centres]
        Ns = [r[2] for r in self.means]
        N_errs = [r[2] for r in self.stds]

        plt.figure()
        max_k = max([max(true_ks), max(ks)])
        min_k = min([min(true_ks), min(ks)])
        plt.errorbar(true_ks, ks, yerr=k_errs, label="k1", linestyle="None", marker="o")
        x = np.linspace(min_k, max_k, 100)
        plt.plot(x, x, "r--")
        plt.xlabel(r"True $k$")
        plt.ylabel(r"recovered $k$")
        plt.title(f"{self.dets} detections")
        plt.savefig(f"{self.dets}_ks.pdf")
        plt.savefig(f"{self.dets}_ks.png")
        plt.figure()

        plt.figure()
        max_k = max([max(true_k2s), max(k2s)])
        min_k = min([min(true_k2s), min(k2s)])
        plt.errorbar(
            true_k2s, k2s, yerr=k2_errs, label="k2", linestyle="None", marker="o"
        )
        x = np.linspace(min_k, max_k, 100)
        plt.plot(x, x, "r--")
        plt.xlabel(r"True $k_w$")
        plt.ylabel(r"recovered $k_w$")
        plt.title(f"{self.dets} detections")
        plt.savefig(f"{self.dets}_k2s.pdf")
        plt.savefig(f"{self.dets}_k2s.png")
        plt.figure()

        max_N = max([max(true_Ns), max(Ns)])
        x = np.linspace(0, max_N, 100)
        plt.errorbar(true_Ns, Ns, yerr=N_errs, label="N", linestyle="None", marker="o")
        # log log plot
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("True N")
        plt.ylabel("recovered N")
        plt.title(f"{self.dets} detections")
        plt.plot(x, x, "r--")
        plt.savefig(f"{self.dets}_exp_Ns.pdf")
        plt.savefig(f"{self.dets}_exp_Ns.png")
        # plt.scatter(true_Ns,N_ratios,label="N")
        plt.legend()

    def plot_fit_exp(self):
        import statistics_basic

        for f, m in zip(self.filename, self.means):
            # get the mu and sigma from the filename
            split = f.split("_")
            # join all but the last element
            yaml_file = "_".join(split[:-1]) + ".yaml"
            real_det = "_".join(split[:-1]) + ".dill"
            # load the yaml file
            with open(yaml_file) as yamlf:
                data = yaml.safe_load(yamlf)
            detection_curve = data["detection_curve"]
            snr_thresh = data["snr_thresh"]
            # load the detection curve
            snr_thresh = statistics_basic.load_detection_fn(
                detection_curve, min_snr_cutoff=snr_thresh
            )
            # print(snr_thresh)
            import statistics_exp
            from bayes_factor_NS_LN_no_a_single import process_detection_results

            # load the detections
            det_fluence, det_width, det_snr, noise_std = process_detection_results(
                real_det
            )
            det_snr = det_snr[det_snr > snr_thresh]
            fit_x = np.linspace(1e-9, np.max(det_snr) * 2, 10000)
            k = m[0]
            det_error = statistics_exp.det_error
            fit_y, p_det, conv_amp_array, conv = statistics_exp.first_exp_plot(
                fit_x, k, det_error
            )
            # fit_y = smoothTriangle(fit_y, 100)
            fit_y = fit_y / np.trapz(fit_y, fit_x)
            fig, ax = plt.subplots(1, 1)
            ax.hist(det_snr, bins="auto", density=True, label=r"$S_{det}$")
            ax.plot(fit_x, fit_y, label="fit", linewidth=2)
            ax.set_xlim(-(np.max(det_snr) * 0.01), np.max(det_snr) * 1.2)
            ax.set_ylim(0, 1.2 * np.max(fit_y))

            # ax.set_xlim(-5, 5)
            # ax.plot(conv_amp_array, conv, label="convolution")
            ax2 = ax.twinx()
            ax2.set_ylim(0, 1.01)
            ax2.plot(
                conv_amp_array,
                p_det,
                label=r"P(det|$S_{det}$)",
                color="b",
                linewidth=2,
                alpha=0.6,
            )
            ax2.set_xlabel(r"$S_{det}$")
            ax2.set_ylabel(r"P(det|$S_{det}$)")
            ax.set_ylabel("Probability density")
            ax.legend(loc=1)
            ax2.legend(loc=2)
            plt.savefig(f"{f}_exp_fit.pdf")
            plt.savefig(f"{f}_exp_fit.png")
            # plt.show()

    def plot_fit_logn(self):
        import statistics_basic

        for f, m in zip(self.filename, self.means):
            # get the mu and sigma from the filename
            split = f.split("_")
            # join all but the last element
            yaml_file = "_".join(split[:-1]) + ".yaml"
            real_det = "_".join(split[:-1]) + ".dill"
            # load the yaml file
            with open(yaml_file) as yamlf:
                data = yaml.safe_load(yamlf)
            detection_curve = data["detection_curve"]
            snr_thresh = data["snr_thresh"]
            # load the detection curve
            snr_thresh = statistics_basic.load_detection_fn(
                detection_curve, min_snr_cutoff=snr_thresh
            )
            # print(snr_thresh)
            import statistics
            from bayes_factor_NS_LN_no_a_single import process_detection_results

            # load the detections
            det_fluence, det_width, det_snr, noise_std = process_detection_results(
                real_det
            )
            det_snr = det_snr[det_snr > snr_thresh]
            fit_x = np.linspace(1e-9, np.max(det_snr) * 2, 10000)
            mu = m[0]
            std = m[1]
            det_error = statistics.det_error
            fit_y, p_det, conv_amp_array, conv = statistics.first_plot(
                fit_x, mu, std, det_error, a=0
            )
            # fit_y = smoothTriangle(fit_y, 100)
            fit_y = fit_y / np.trapz(fit_y, fit_x)
            fig, ax = plt.subplots(1, 1)
            ax.hist(det_snr, bins="auto", density=True, label=r"$S_{det}$")
            ax.plot(fit_x, fit_y, label="fit", linewidth=2)
            ax.set_xlim(-(np.max(det_snr) * 0.01), np.max(det_snr) * 1.2)
            ax.set_ylim(0, 1.2 * np.max(fit_y))

            # ax.set_xlim(-5, 5)
            # ax.plot(conv_amp_array, conv, label="convolution")
            ax2 = ax.twinx()
            ax2.set_ylim(0, 1.01)
            ax2.plot(
                conv_amp_array,
                p_det,
                label=r"P(det|$S_{det}$)",
                color="b",
                linewidth=2,
                alpha=0.6,
            )
            ax2.set_xlabel("SNR")
            ax2.set_ylabel(r"P(det|$S_{det}$)")
            ax.set_ylabel("Probability density")
            ax.legend(loc=1)
            ax2.legend(loc=2)
            plt.savefig(f"{f}_logn_fit.pdf")
            plt.savefig(f"{f}_logn_fit.png")
            # plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", nargs="+", help="filenames to plot")
    parser.add_argument("-p", help="period", type=float, default=-1)
    parser.add_argument("--obs_time", help="observation time", type=float, default=1)
    parser.add_argument(
        "--plot_accuracy", help="plot the accuracy of the results", action="store_true"
    )
    parser.add_argument(
        "--exp",
        help="making the plots for an exponential distribution",
        action="store_true",
    )
    parser.add_argument(
        "--bayes_ratio", help="plot the bayes ratio", action="store_true"
    )

    args = parser.parse_args()
    global period
    global obs_time
    period = args.p
    obs_time = args.obs_time
    plot_accuracy = args.plot_accuracy
    filenames = args.filenames
    dp = dynesty_plot(filenames)
    dp.load_filenames()
    if args.exp:
        dp.plot_corner(labels=[r"k1", "k2", "N"], plot=False)
        # dp.plot_fit_exp()
        if plot_accuracy:
            dp.plot_accuracy_exp()
        if args.bayes_ratio:
            dp.plot_bayes_ratio_exp()

    else:
        dp.plot_corner(labels=[r"$\mu$", r"$\sigma$", "N"], plot=False)
        # dp.plot_fit_logn()
        if plot_accuracy:
            dp.plot_accuracy_logn()
        if args.bayes_ratio:
            dp.plot_bayes_ratio_logn()
