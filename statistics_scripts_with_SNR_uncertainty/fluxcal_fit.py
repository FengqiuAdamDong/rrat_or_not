import matplotlib.pyplot as plt
import numpy as np
import argparse
import glob
from chimepsr_fluxcal.utils.sefd import SEFD
from dynesty.utils import quantile
import smplotlib
from dynesty.utils import mean_and_cov
import copy
import csv

# sefd = SEFD()
# sefd.set_pointing(ra_deg, dec_deg, mjd= hdr.tstart+(total_time/2/86400))
# sefd.compute_sefd()


def calibrator_average(calibrator_name):
    calibrator_files = glob.glob(calibrator_name + "*.npz")
    t_telescope_arr = []
    for calibrator_file in calibrator_files:
        calibrator_data = np.load(calibrator_file, allow_pickle=True)["sefd"].tolist()
        t_telescope = calibrator_data.calibrated_tstruct_rcvr
        t_telescope_arr.append(t_telescope)
        # plt.plot(calibrator_data.freqs, t_telescope, label='t_telescope')
        # plt.plot(calibrator_data.freqs, calibrator_data.tsky, label='tsky')
        # plt.plot(calibrator_data.freqs, calibrator_data.calibrated_sefd_tstruct_rcvr, label='calibrated_sefd_tstruct_rcvr')
        # plt.legend()

        # plt.show()
    t_telescope_arr = np.array(t_telescope_arr)
    t_telescope_mean = np.mean(t_telescope_arr, axis=0)
    # import pdb; pdb.set_trace()
    # plt.plot(calibrator_data.freqs, t_telescope_mean)
    # plt.show()
    return t_telescope_mean


def calibrate_snr(bayes_fit, sefd, cal_error):
    # cal error is relative i. 10% on the sefd
    samples = bayes_fit.samples
    importance_weights = bayes_fit.importance_weights()
    multiplicative_factor = np.mean(sefd.calibrated_sefd_tstruct_rcvr) / np.sqrt(
        400e6 * 327.68e-6 * 2
    )  # units of Jy
    # additive factor for log normal mean
    calibrated_samples = copy.deepcopy(samples)
    calibrated_samples_upper = copy.deepcopy(calibrated_samples)
    calibrated_samples_lower = copy.deepcopy(calibrated_samples)

    calibrated_samples[:, 0] = samples[:, 0] + np.log(multiplicative_factor)
    calibrated_samples_upper[:, 0] = calibrated_samples_upper[:, 0] + np.log(
        multiplicative_factor * (1 + cal_error)
    )
    calibrated_samples_lower[:, 0] = calibrated_samples_lower[:, 0] + np.log(
        multiplicative_factor * (1 - cal_error)
    )
    quantiles = []
    quantiles_upper = []
    quantiles_lower = []
    for i in range(samples.shape[1]):
        quantiles.append(
            quantile(calibrated_samples[:, i], [0.159, 0.5, 0.841], weights=importance_weights)
        )
        quantiles_upper.append(
            quantile(calibrated_samples_upper[:, i], [0.159, 0.5, 0.841], weights=importance_weights)
        )
        quantiles_lower.append(
            quantile(calibrated_samples_lower[:, i], [0.159, 0.5, 0.841], weights=importance_weights)
        )
    quantiles = np.array(quantiles)
    quantiles_upper = np.array(quantiles_upper)
    quantiles_lower = np.array(quantiles_lower)
    mean, cov = mean_and_cov(calibrated_samples, weights=importance_weights)
    return (
        calibrated_samples,
        calibrated_samples_upper,
        calibrated_samples_lower,
        quantiles,
        mean,
        cov,
    )


def fluxcal_fit(
    bayes_fit_npz, calibrator_name, ra, dec, transit_time, cal_error=0.153, plot=True
):
    # find all the calibrator npz files
    sefd = SEFD()
    sefd.set_pointing(ra, dec, mjd=transit_time)
    sefd.compute_sefd()

    # find all the npz files starting with the calibrator name
    t_telescope_mean = calibrator_average(calibrator_name)
    # construct the SEFD object
    sefd.calibrated_tstruct_rcvr = t_telescope_mean
    sefd.calibrated_sefd_tstruct_rcvr = (t_telescope_mean + sefd.tsky) / sefd.gain
    # plot the temperatures etc
    if plot:
        fig, ax1 = plt.subplots()
        ax1.plot(sefd.freqs, sefd.calibrated_tstruct_rcvr, label="t_telescope")
        ax1.plot(sefd.freqs, sefd.tsky, label="tsky")
        ax1.plot(
            sefd.freqs,
            sefd.calibrated_sefd_tstruct_rcvr,
            label="calibrated_sefd_tstruct_rcvr",
        )
        # plot the gain on a second axis
        ax2 = plt.gca().twinx()
        ax2.plot(sefd.freqs, sefd.gain, color="red", label="Gain")
        ax1.legend()
        ax2.legend()
        ax1.set_xlabel("Frequency (MHz)")
        ax1.set_ylabel("Temperature (K) or SEFD (Jy)")
        ax2.set_ylabel("Gain (K/Jy)")
        plt.savefig(
            f'{bayes_fit_npz.split("/")[-1].split(".")[0]}_{calibrator_name}.png'
        )
    # load the bayes fit npz file
    bayes_fit = np.load(bayes_fit_npz, allow_pickle=True)["results"].tolist()

    (
        calibrated_samples,
        calibrated_samples_upper,
        calibrated_samples_lower,
        quantiles,
        mean,
        cov,
    ) = calibrate_snr(bayes_fit, sefd, cal_error)
    print(mean)
    print(quantiles)
    np.savez(
        f'{bayes_fit_npz.split("/")[-1].split(".")[0]}_{calibrator_name}_calibrated.npz',
        results=bayes_fit,
        quantiles=quantiles,
        calibrated_samples=calibrated_samples,
        calibrated_samples_upper=calibrated_samples_upper,
        calibrated_samples_lower=calibrated_samples_lower,
        mean=mean,
        cov=cov,
    )
    # assuming only the log normal distribution


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit the flux calibrator data")
    parser.add_argument("-bayes_fit_npz", type=str, help="Bayes fit npz file")
    parser.add_argument("-calibrator_name", type=str, help="Calibrator name")
    parser.add_argument("-ra", type=float, help="RA of the source")
    parser.add_argument("-dec", type=float, help="DEC of the source")
    parser.add_argument(
        "-transit_time", type=float, help="Transit time of the source in MJD"
    )
    parser.add_argument(
        "-cal_error", type=float, help="relative calibration error", required=True
    )

    args = parser.parse_args()
    fluxcal_fit(
        args.bayes_fit_npz,
        args.calibrator_name,
        args.ra,
        args.dec,
        args.transit_time,
        args.cal_error,
        plot=False,
    )
