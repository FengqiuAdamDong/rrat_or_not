import numpy as np
import matplotlib.pyplot as plt
import dynesty
from bayes_factor_NS_LN_no_a_single import loglikelihood
from bayes_factor_NS_LN_no_a_single import pt_Uniform_N
from dynesty import plotting as dyplot
import smplotlib
import yaml
from statistics import statistics_ln


class post_process:
    def __init__(self, fns):
        self.fns = fns
        for fn in fns:
            if "expexp" in fn:
                self.expexpfn = fn
            elif "lnln" in fn:
                self.lnlnfn = fn
            elif "expln" in fn:
                self.explnfn = fn
            elif "lnexp" in fn:
                self.lnexpfn = fn
        # load data
        self.expexp_results = self.load_data(self.expexpfn)
        self.lnln_results = self.load_data(self.lnlnfn)
        self.expln_results = self.load_data(self.explnfn)
        self.lnexp_results = self.load_data(self.lnexpfn)

    def load_data(self, fn):
        data = np.load(fn, allow_pickle=True)["results"].tolist()

        return data

    def plot_corner(
        self,
    ):
        label_expexp = ["k1", "k2", "N"]
        fig, axes = dyplot.cornerplot(self.expexp_results, labels=label_expexp)
        plt.savefig("expexp_corner.png")
        plt.close()
        label_lnln = ["mu1", "std1", "mu2", "std2", "N"]
        fig, axes = dyplot.cornerplot(self.lnln_results, labels=label_lnln)
        plt.savefig("lnln_corner.png")
        plt.close()
        label_expln = ["k1", "mu2", "std2", "N"]
        fig, axes = dyplot.cornerplot(self.expln_results, labels=label_expln)
        plt.savefig("expln_corner.png")
        plt.close()
        label_lnexp = ["mu1", "std1", "k2", "N"]
        fig, axes = dyplot.cornerplot(self.lnexp_results, labels=label_lnexp)
        plt.savefig("lnexp_corner.png")
        plt.close()

    def plot_bayes_ratio(
        self,
    ):
        evidence_expexp = self.expexp_results["logz"][-1]
        evidence_expexp_err = self.expexp_results["logzerr"][-1]
        evidence_lnln = self.lnln_results["logz"][-1]
        evidence_lnln_err = self.lnln_results["logzerr"][-1]
        evidence_expln = self.expln_results["logz"][-1]
        evidence_expln_err = self.expln_results["logzerr"][-1]
        evidence_lnexp = self.lnexp_results["logz"][-1]
        evidence_lnexp_err = self.lnexp_results["logzerr"][-1]

        plt.errorbar(
            [0, 1, 2, 3],
            [evidence_expexp, evidence_lnln, evidence_expln, evidence_lnexp],
            yerr=[
                evidence_expexp_err,
                evidence_lnln_err,
                evidence_expln_err,
                evidence_lnexp_err,
            ],
            fmt="o",
        )
        plt.xticks([0, 1, 2, 3], ["expexp", "lnln", "expln", "lnexp"])
        plt.ylabel("log evidence")
        plt.savefig("log_evidence.png")
        plt.close()

    def get_best_fit_values(self, dynesty_results):
        from dynesty.utils import quantile

        samples = dynesty_results.samples
        importance_weights = dynesty_results.importance_weights()
        quantiles = []
        for i in range(samples.shape[1]):
            quantiles.append(
                quantile(samples[:, i], [0.159, 0.5, 0.841], weights=importance_weights)
            )
        quantiles = np.array(quantiles)
        from dynesty.utils import mean_and_cov

        mean, cov = mean_and_cov(samples, weights=importance_weights)
        return quantiles, mean, cov

    def get_best_fit_values_all(
        self,
    ):
        (
            self.expexp_quantiles,
            self.expexp_mean,
            self.expexp_cov,
        ) = self.get_best_fit_values(self.expexp_results)
        self.lnln_quantiles, self.lnln_mean, self.lnln_cov = self.get_best_fit_values(
            self.lnln_results
        )
        (
            self.expln_quantiles,
            self.expln_mean,
            self.expln_cov,
        ) = self.get_best_fit_values(self.expln_results)
        (
            self.lnexp_quantiles,
            self.lnexp_mean,
            self.lnexp_cov,
        ) = self.get_best_fit_values(self.lnexp_results)

    def plot_fit(self, yaml_file):
        # load yaml file
        import cupy as cp
        from bayes_factor_LNLN import process_detection_results

        with open(yaml_file, "r") as stream:
            try:
                yaml_data = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        dill_file = yaml_file.replace("yaml", "dill")
        snr_thresh = yaml_data["snr_thresh"]
        width_thresh = yaml_data["width_thresh"]
        detection_curve = yaml_data["detection_curve"]
        flux_cal = 1
        likelihood_calc = statistics_ln(
            detection_curve,
            plot=True,
            flux_cal=flux_cal,
            snr_cutoff=snr_thresh,
            width_cutoff=width_thresh,
        )
        (
            det_fluence,
            det_width,
            det_snr,
            noise_std,
            likelihood_calc,
            low_width_flag,
            logn_lower,
        ) = process_detection_results(
            dill_file, snr_thresh, width_thresh, likelihood_calc
        )

        snr_array = np.linspace(0, 80, 1000)
        width_array = np.linspace(0, 30, 1001) * 1e-3
        # fits to plot
        best_fit_arr = [
            self.expexp_quantiles,
            self.lnln_quantiles,
            self.expln_quantiles,
            self.lnexp_quantiles,
        ]
        fit_type_arr = [["exp", "exp"], ["ln", "ln"], ["exp", "ln"], ["ln", "exp"]]

        sigma_snr = likelihood_calc.detected_error_snr
        sigma_width = likelihood_calc.detected_error_width
        for best_fit_vals, fit_type in zip(best_fit_arr, fit_type_arr):
            likelihood = np.zeros((len(snr_array), len(width_array)))
            for i, snr in enumerate(snr_array):
                snr_array_temp = np.ones_like(width_array) * snr
                width_array_temp = width_array
                snr_array_temp = cp.asarray(snr_array_temp)
                width_array_temp = cp.asarray(width_array_temp)
                likelihood_calc.calculate_pdet(
                    snr_array_temp, width_array_temp, filter=False
                )

                if (fit_type[0] == "exp") & (fit_type[1] == "exp"):
                    mu_ln = best_fit_vals[0][1]
                    std_ln = 0
                    w_mu_ln = best_fit_vals[1][1]
                    w_std_ln = 0
                if (fit_type[0] == "ln") & (fit_type[1] == "ln"):
                    mu_ln = best_fit_vals[0][1]
                    std_ln = best_fit_vals[1][1]
                    w_mu_ln = best_fit_vals[2][1]
                    w_std_ln = best_fit_vals[3][1]
                if (fit_type[0] == "exp") & (fit_type[1] == "ln"):
                    mu_ln = best_fit_vals[0][1]
                    std_ln = 0
                    w_mu_ln = best_fit_vals[1][1]
                    w_std_ln = best_fit_vals[2][1]
                if (fit_type[0] == "ln") & (fit_type[1] == "exp"):
                    mu_ln = best_fit_vals[0][1]
                    std_ln = best_fit_vals[1][1]
                    w_mu_ln = best_fit_vals[2][1]
                    w_std_ln = 0

                loglike_sum, loglike_all = likelihood_calc.first_cupy_plot(
                    snr_array_temp,
                    width_array_temp,
                    mu_ln,
                    std_ln,
                    w_mu_ln,
                    w_std_ln,
                    sigma_amp=sigma_snr,
                    sigma_w=sigma_width,
                    a=0,
                    lower_c=0,
                    upper_c=cp.inf,
                    amp_dist=fit_type[0],
                    w_dist=fit_type[1],
                )
                likelihood[i, :] = np.exp(loglike_all.get())
            likelihood_norm = likelihood / np.trapz(
                np.trapz(likelihood, snr_array, axis=0), width_array
            )
            likelihood_norm = likelihood_norm.T
            fig, ax = plt.subplots(1, 2, figsize=(15, 7))
            ax[0].hist2d(det_snr, det_width * 1e3, bins=50, density=True)
            ax[0].set_xlabel("SNR")
            ax[0].set_ylabel("Width (ms)")
            ax[1].pcolormesh(snr_array, width_array * 1e3, likelihood_norm)
            ax[1].set_xlabel("SNR")
            ax[1].set_ylabel("Width (ms)")
            # set the same limits as ax[0]
            ax[1].set_xlim(ax[0].get_xlim())
            ax[1].set_ylim(ax[0].get_ylim())
            plt.savefig(f"likelihood_{fit_type[0]}_{fit_type[1]}.png")

            # marginalise over the dimensions
            marg_snr = np.trapz(likelihood_norm, width_array, axis=0)
            marg_width = np.trapz(likelihood_norm, snr_array, axis=1)
            fig, ax = plt.subplots(1, 2, figsize=(15, 7))
            ax[0].plot(snr_array, marg_snr)
            ax[0].hist(det_snr, bins="auto", density=True)
            ax[0].set_xlabel("SNR")
            ax[0].set_ylabel("Probability")
            ax[0].set_xlim(0, max(det_snr))
            ax[1].plot(width_array * 1e3, marg_width / 1e3)
            ax[1].hist(det_width * 1e3, bins="auto", density=True)
            ax[1].set_xlabel("Width (ms)")
            ax[1].set_ylabel("Probability")
            ax[1].set_xlim(0, max(det_width * 1e3))
            plt.savefig(f"marginal_{fit_type[0]}_{fit_type[1]}.png")
            plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("fn", nargs="+", help="filename")
    parser.add_argument(
        "--plot_fit",
        type=str,
        default="",
        help="plot fit, defaults to nothing and not plot fit, must give yaml file",
    )
    args = parser.parse_args()
    fns = args.fn
    yaml_file = args.plot_fit

    pp = post_process(fns)
    pp.get_best_fit_values_all()
    if yaml_file != "":
        pp.plot_fit(yaml_file)
    pp.plot_corner()
    pp.plot_bayes_ratio()
