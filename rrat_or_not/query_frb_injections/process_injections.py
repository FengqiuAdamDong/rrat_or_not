import numpy as np
import matplotlib.pyplot as plt
from rrat_or_not.query_frb_injections.load_injections_data import injectionsData
import dill
from scipy.optimize import basinhopping

from scipy.stats import norm

# from rrat_or_not.injection_scripts_fluence.injection_stats import inject_stats
from scipy import interpolate as interp


def gen_log(x, B, v, K, M, cutoff):
    y = K / ((1 + np.exp(-B * (x - M))) ** (1 / v))
    y[x < cutoff] = 0
    return y


def flipped_gen_log(x, B, v, K, M, cutoff):
    # invert the gen_log function
    y = K - gen_log(x, B, v, K, M, 0)
    y[x > cutoff] = 0
    return y


def gen_log_2d(mesh_amp, mesh_width, K_amp, K_width, cutoff_amp, cutoff_width):
    # K_amp are the amplitude parameters, and K_width are the width parameters
    # y = gen_log(mesh_amp, K_amp[0], K_amp[1], K_amp[2], K_amp[3], cutoff_amp) * \
    # flipped_gen_log(mesh_width, K_width[0], K_width[1], K_width[2], K_width[3], cutoff_width)
    # make y a product of gen log and a 4th order polynomial
    y = gen_log(
        mesh_amp, K_amp[0], K_amp[1], K_amp[2], K_amp[3], cutoff_amp
    ) * np.polyval([K_width[0], K_width[1], K_width[2], K_width[3]], mesh_width)
    y[y < 0] = 0
    y[y > 1] = 1
    return y


def forward_model(x, k1, k2, k3, x0, cutoff):
    # this is just a wrapper function so that I only need to change this one reference to change the function used
    return gen_log(x, k1, k2, k3, x0, cutoff)
    # return piecewise_tanh(x, k1, k2, x0, cutoff)
    # return piecewise_logistic(x, k1, k2, x0, cutoff)


def get_formed_beam():
    from beam_model import formed

    formed_beam_model = formed.FFTFormedActualBeamModel()
    beam_id_base = np.arange(0, 256)
    freqs = np.array([600])
    beam_x = []
    beam_y = []
    for i in range(4):
        beam_ids = beam_id_base + (i * 1000)
        beam_positions = formed_beam_model.get_beam_positions(beam_ids, freqs)
        for pos in beam_positions:
            beam_x.append(pos[0][0])
            beam_y.append(pos[0][1])

    return np.min(beam_x), np.max(beam_x), np.min(beam_y), np.max(beam_y)


# inherit from injectionsData to add processed attributes
class injectionsData_processed(injectionsData):
    def __init__(self, injection_dict, detection_dict=None):
        super().__init__(injection_dict, detection_dict)

    # overload with a second method where you can just import an injectionsData object
    @classmethod
    def from_injectionsData(cls, inj_data_obj):
        return cls(inj_data_obj.injection, inj_data_obj.detection)

    def process_injections_data(self):
        self.injection_dm = self.injection.get("dm", None)
        self.injection_fluence_jy_ms = self.injection.get("fluence_jy_ms", None)
        self.injection_pulse_width_ms = self.injection.get("pulse_width_ms", None)
        self.injection_tau_1_ghz_ms = self.injection.get(
            "extra_injection_parameters", {}
        ).get("tau_1_ghz_ms", None)
        self.beam_x = self.injection.get("extra_injection_parameters", {}).get(
            "beam_x", None
        )
        self.beam_y = self.injection.get("extra_injection_parameters", {}).get(
            "beam_y", None
        )

        self.detected = False

        if self.detection is not None:
            self.detection_snr = self.detection.get("combined_snr", None)
            self.detection_dm = self.detection.get("dm", None)
            self.detected = True
        else:
            self.detection_snr = None
            self.detection_dm = None

        # delete the injection and detection dicts to save memory and storage
        del self.injection
        del self.detection


class selection_fluence_width:
    def __init__(self, injections_data_arr):
        self.injections_data_arr = injections_data_arr
        self.load_tau_width_to_effective_width_map()

    def load_tau_width_to_effective_width_map(self, npz_file="effective_widths.npz"):
        data = np.load(npz_file)
        taus = data["taus"]
        sigmas = data["sigmas"]
        # first axis is sigma, second axis is tau
        self.mtaus = data["mtaus"]
        self.msigmas = data["msigmas"]
        self.effective_widths = data["effective_widths"]

    def interpolate_effective_width(self, tau, sigma):
        interpolator = interp.RegularGridInterpolator(
            (self.mtaus[0, :], self.msigmas[:, 0]),
            self.effective_widths.T,
            bounds_error=False,
            fill_value=None,
        )
        point = np.array([[tau, sigma]])
        effective_width = interpolator(point)[0]
        return effective_width

    def test_selection(self):
        injection_dm = np.array([obj.injection_dm for obj in self.injections_data_arr])
        detection_dm = np.array([obj.detection_dm for obj in self.injections_data_arr])
        plt.figure()
        plt.scatter(injection_dm, detection_dm, c="blue", alpha=0.5)
        plt.xlabel("Injection DM")
        plt.ylabel("Detection DM")
        plt.title("Injection DM vs Detection DM")
        plt.show()

    def bin_fluence_dm(self):
        fluences = np.array(
            [obj.injection_fluence_jy_ms for obj in self.injections_data_arr]
        )
        pulse_width_ms = np.array(
            [obj.injection_pulse_width_ms for obj in self.injections_data_arr]
        )
        tau_1_ghz_ms = np.array(
            [obj.injection_tau_1_ghz_ms for obj in self.injections_data_arr]
        )
        # convert this to scattering timescale at 600mhz
        tau_600_mhz_ms = tau_1_ghz_ms * (1000 / 600) ** 4
        # calculate the effective width by interpolating the npzfile
        effective_width = np.array(
            [
                self.interpolate_effective_width(tau, pw)
                for tau, pw in zip(tau_600_mhz_ms, pulse_width_ms)
            ]
        )

        print("min max tau 600 mhz", np.min(tau_600_mhz_ms), np.max(tau_600_mhz_ms))
        print("min max width", np.min(pulse_width_ms), np.max(pulse_width_ms))
        detected = np.array([obj.detected for obj in self.injections_data_arr])
        # make a 2d histogram of fluence vs pulse width, color coded by detection fraction
        # just go out to 1000 for fluences
        fluence_bins = np.logspace(
            np.log10(np.min(fluences[fluences > 0])), np.log10(1000), 10
        )
        # make 11 bins in width and 10 in fluence so that it's easier to track
        effective_width_bins = np.logspace(np.log10(1), np.log10(50), 11)
        pulse_width_bins = np.logspace(np.log10(1), np.log10(50), 11)
        tau_600_mhz_ms_bins = np.logspace(np.log10(1), np.log10(50), 11)
        # plot a 2d histogram of fluence vs pulse width, color coded by effective width
        effective_width_av = np.zeros(
            (len(tau_600_mhz_ms_bins) - 1, len(pulse_width_bins) - 1)
        )
        for i in range(len(tau_600_mhz_ms_bins) - 1):
            for j in range(len(pulse_width_bins) - 1):
                in_bin = (
                    (tau_600_mhz_ms >= tau_600_mhz_ms_bins[i])
                    & (tau_600_mhz_ms < tau_600_mhz_ms_bins[i + 1])
                    & (pulse_width_ms >= pulse_width_bins[j])
                    & (pulse_width_ms < pulse_width_bins[j + 1])
                )
                if np.sum(in_bin) > 0:
                    # average all the in_bin effective widths
                    effective_width_av[i, j] = np.mean(effective_width[in_bin])
                else:
                    effective_width_av[i, j] = np.nan
        # These all load the default best-fit model (see model-selection.ipynb)
        #
        detection_fraction = np.zeros(
            (len(fluence_bins) - 1, len(pulse_width_bins) - 1)
        )
        for i in range(len(fluence_bins) - 1):
            for j in range(len(effective_width_bins) - 1):
                in_bin = (
                    (fluences >= fluence_bins[i])
                    & (fluences < fluence_bins[i + 1])
                    & (effective_width >= effective_width_bins[j])
                    & (effective_width < effective_width_bins[j + 1])
                )
                total_in_bin = np.sum(in_bin)
                if total_in_bin > 0:
                    detected_in_bin = np.sum(detected[in_bin])
                    detection_fraction[i, j] = detected_in_bin / total_in_bin
                else:
                    detection_fraction[i, j] = np.nan

        self.detection_fraction = detection_fraction
        self.det_frac_matrix_snr = self.detection_fraction

        self.effective_width_bins = effective_width_bins
        self.fluence_bins = fluence_bins
        # set to midpoints of the bin edges
        self.unique_widths = np.array(
            [
                0.5 * (effective_width_bins[i] + effective_width_bins[i + 1])
                for i in range(len(effective_width_bins) - 1)
            ]
        )
        # self.fluence_bin_edges = fluence_bin_edges
        # set this as unique snrs too
        self.unique_snrs = np.array(
            [
                0.5 * (fluence_bins[i] + fluence_bins[i + 1])
                for i in range(len(fluence_bins) - 1)
            ]
        )
        self.unique_amplitude = self.unique_snrs

        # save self
        with open("temp.dill", "wb") as of:
            dill.dump(self, of)

        plt.figure()
        plt.pcolormesh(
            tau_600_mhz_ms_bins,
            pulse_width_bins,
            effective_width_av.T,
            shading="auto",
            cmap="viridis",
        )
        plt.xlabel("Tau 1 GHz (ms)")
        plt.ylabel("Pulse Width (ms)")
        plt.colorbar(label="Average Effective Width (ms)")
        plt.xscale("log")
        plt.yscale("log")
        plt.savefig("effective_width_vs_tau_width.png")

        plt.figure()
        plt.pcolormesh(
            fluence_bins,
            effective_width_bins,
            detection_fraction.T,
            shading="auto",
            cmap="viridis",
        )
        plt.xlabel("Fluence (Jy ms)")
        plt.ylabel("Pulse Width (ms)")
        plt.colorbar(label="Detection Fraction")
        plt.xscale("log")
        plt.yscale("log")
        plt.title("Detection Fraction vs Fluence and Pulse Width")
        plt.savefig("detection_fraction_vs_fluence_width.png")
        # plot some slices of the selection function in fluence at fixed pulse widths
        plt.figure(figsize=(10, 8))
        fluence_axis = 0.5 * (fluence_bins[:-1] + fluence_bins[1:])
        i = 0
        while i < len(effective_width_bins) - 1:
            # do every 10th pulse width bin
            mid_pulse_width = 0.5 * (
                effective_width_bins[i] + effective_width_bins[i + 1]
            )
            plt.plot(fluence_axis, detection_fraction[:, i], label=f"{mid_pulse_width}")
            i += 1
            # ax[j].set_xscale('log')
            # ax[j].set_yscale('log')
            # set the x axis from 0-50
        plt.xscale("log")
        plt.ylabel("Selection Probability")
        plt.xlabel("Fluence (Jy ms)")
        plt.legend()
        plt.savefig("selection_function_slices_fluence.png")

    def forward_model_amp_det(self):
        karr = []
        self.forward_model_cutoffs = []
        for i in range(len(self.unique_widths)):
            snrs = self.unique_snrs
            snrs = np.array(snrs)
            det_fracs = self.det_frac_matrix_snr[:, i]
            det_fracs = np.array(det_fracs)
            # determine where x0 is
            snr_interp_arr = np.logspace(np.log10(min(snrs)), np.log10(max(snrs)), 1000)
            interp_det_fracs = np.interp(snr_interp_arr, snrs, det_fracs)
            # find where it's closest to 0.5
            x0 = snr_interp_arr[np.argmin(np.abs(interp_det_fracs - 0.5))]

            def p_det_st(x, k1, k2, k3, x0, det_err, cutoff):
                sdet = np.linspace(min(x) - 3 * det_err, max(x) + 3 * det_err, 1000)
                sdet_giv_st = norm.pdf(
                    sdet, loc=x[np.newaxis, :], scale=det_err[np.newaxis, :]
                )
                pdet_giv_sdet = forward_model(sdet, k1, k2, k3, x0, cutoff)
                integral = np.trapezoid(sdet_giv_st * pdet_giv_sdet, sdet, axis=0)
                return integral

            def loglike(X, snr_arr, det_fracs, det_err, cutoff):
                sigma = X[4]
                # use a gaussian likelihood
                loglike = np.nansum(
                    -0.5
                    * (
                        p_det_st(snr_arr, X[0], X[1], X[2], X[3], det_err, cutoff)
                        - det_fracs
                    )
                    ** 2
                    / sigma**2
                    - np.log(sigma * np.sqrt(2 * np.pi))
                )
                return -1 * loglike

            cutoff = np.argwhere(det_fracs < 0.1)
            cutoff = np.max(cutoff)
            self.forward_model_cutoffs.append(snrs[cutoff])
            # this is temporarily set to be from 0.5 to 2.0 linearly spaced
            self.detect_error_snr = np.linspace(0.5, 2.0, len(snrs))
            bounds = [(-50, 50), (-50, 50), (-10, 10), (-1000, 1000), (0.01, 1)]
            args = (snrs, det_fracs, self.detect_error_snr, snrs[cutoff])
            minimizer_kwargs = dict(method="Nelder-Mead", args=args, bounds=bounds)
            init = [1, 1, 1, x0, 0.05]
            res = basinhopping(
                loglike, init, minimizer_kwargs=minimizer_kwargs, niter=50
            )
            # fit the model
            k1, k2, k3, x0, sigma = res.x
            print(
                f"fitted sigma {sigma} k1 {k1} k2 {k2} k3 {k3} x0 {x0} cutoff {snrs[cutoff]} width {self.unique_widths[i]}"
            )
            karr.append(res.x)
            plt.figure()
            plt.plot(snrs, det_fracs, "o", label="Data Detection Fraction|True")
            plt.plot(
                snrs,
                p_det_st(snrs, k1, k2, k3, x0, self.detect_error_snr, snrs[cutoff]),
                label="Fitted Model",
            )
            # if I use interped array, I also need an interped error
            interp_err = np.interp(snr_interp_arr, snrs, self.detect_error_snr)
            plt.plot(
                snr_interp_arr,
                p_det_st(snr_interp_arr, k1, k2, k3, x0, interp_err, snrs[cutoff]),
                label="Fitted Model Smooth",
            )
            plt.plot(
                snr_interp_arr,
                forward_model(snr_interp_arr, k1, k2, k3, x0, snrs[cutoff]),
                label="det|sdet",
            )
            # set x axis to log
            plt.xscale("log")
            plt.legend()
            plt.savefig(f"forward_model_fit_width_{self.unique_widths[i]:.2f}.png")
            plt.close()

        self.karr_amp = karr
        with open("temp_amp_fitted.dill", "wb") as of:
            dill.dump(self, of)
    
    def plot_modelled_selection_effects(self):
        modelled_selection = np.zeros((1000, len(self.unique_widths)))
        amps = np.linspace(np.min(self.unique_snrs), np.max(self.unique_snrs), 1000)
        widths = self.unique_widths
        for i in range(len(self.unique_widths)):
            p_det_sdet = forward_model(
                amps,
                self.karr_amp[i][0],
                self.karr_amp[i][1],
                self.karr_amp[i][2],
                self.karr_amp[i][3],
                self.forward_model_cutoffs[i],
            )
            modelled_selection[:, i] = p_det_sdet
        plt.figure()
        plt.pcolormesh(
            np.log10(amps),
            np.log10(widths),
            modelled_selection.T,
        )
        plt.xlabel("amps")
        plt.ylabel("Pulse Width (ms)")
        plt.colorbar(label="Modeled Selection Probability")
        # plt.xscale("log")
        # plt.yscale("log")
        plt.title("Modeled Selection Probability vs Amplitude and Pulse Width")
        plt.show()






if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process injections data.")
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input .npy file containing injections data array.",
    )
    args = parser.parse_args()
    beam_x_min, beam_x_max, beam_y_min, beam_y_max = get_formed_beam()

    if args.input_file.endswith(".dill"):
        with open(args.input_file, "rb") as f:
            selection = dill.load(f)
    else:
        injections_data_arr = np.load(args.input_file, allow_pickle=True)

        injections_data_obj = [
            injectionsData_processed.from_injectionsData(inj)
            for inj in injections_data_arr
        ]
        for inj_obj in injections_data_obj:
            inj_obj.process_injections_data()
        beam_x_arr = np.array([inj.beam_x for inj in injections_data_obj])
        beam_y_arr = np.array([inj.beam_y for inj in injections_data_obj])
        # only keep those in the formed beam area
        in_beam = (
            (beam_x_arr >= beam_x_min)
            & (beam_x_arr <= beam_x_max)
            & (beam_y_arr >= beam_y_min)
            & (beam_y_arr <= beam_y_max)
        )
        injections_data_obj = [
            inj for i, inj in enumerate(injections_data_obj) if in_beam[i]
        ]
        beam_x_arr = beam_x_arr[in_beam]
        beam_y_arr = beam_y_arr[in_beam]

        plt.figure()
        plt.scatter(beam_x_arr, beam_y_arr, alpha=0.5)
        plt.xlabel("Beam X")
        plt.ylabel("Beam Y")
        plt.show()

        selection = selection_fluence_width(injections_data_obj)
        # selection.test_selection()
        selection.bin_fluence_dm()

    # selection.forward_model_amp_det()
    # selection.plot_modelled_selection_effects()
