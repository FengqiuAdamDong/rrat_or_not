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
        peak_fluxes = fluences / effective_width
            

        print("min max tau 600 mhz ms", np.min(tau_600_mhz_ms), np.max(tau_600_mhz_ms))
        print("min max width ms", np.min(pulse_width_ms), np.max(pulse_width_ms))
        detected = np.array([obj.detected for obj in self.injections_data_arr])
        # make a 2d histogram of fluence vs pulse width, color coded by detection fraction
        # just go out to 1000 for fluences
        fluence_bins = np.logspace(
            np.log10(np.min(fluences[fluences > 0])), np.log10(1000), 10
        )
        peak_flux_bins = np.logspace(
            np.log10(np.min(peak_fluxes[peak_fluxes > 0])), np.log10(1000), 10
        )

        # make 11 bins in width and 10 in fluence so that it's easier to track
        effective_width_bins = np.logspace(np.log10(1), np.log10(50), 11)

        detection_fraction_fluence = np.zeros(
            (len(fluence_bins) - 1, len(effective_width_bins) - 1)
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
                    detection_fraction_fluence[i, j] = detected_in_bin / total_in_bin
                else:
                    detection_fraction_fluence[i, j] = np.nan

        detection_fraction_peak_flux = np.zeros(
            (len(peak_flux_bins) - 1, len(effective_width_bins) - 1)
        )
        for i in range(len(peak_flux_bins) - 1):
            for j in range(len(effective_width_bins) - 1):
                in_bin = (
                    (peak_fluxes >= peak_flux_bins[i])
                    & (peak_fluxes < peak_flux_bins[i + 1])
                    & (effective_width >= effective_width_bins[j])
                    & (effective_width < effective_width_bins[j + 1])
                )
                total_in_bin = np.sum(in_bin)
                if total_in_bin > 0:
                    detected_in_bin = np.sum(detected[in_bin])
                    detection_fraction_peak_flux[i, j] = detected_in_bin / total_in_bin
                else:
                    detection_fraction_peak_flux[i, j] = np.nan

        self.detection_fraction_fluence = detection_fraction_fluence
        self.detection_fraction_peak_flux = detection_fraction_peak_flux

        self.effective_width_bins_ms = effective_width_bins
        self.effective_width_bins = effective_width_bins / 1000

        self.fluence_bins = fluence_bins
        self.peak_flux_bins = peak_flux_bins

        #find the bin midpoints
        self.unique_widths_ms = np.array(
            [
                0.5 * (effective_width_bins[i] + effective_width_bins[i + 1])
                for i in range(len(effective_width_bins) - 1)
            ]
        )
        self.unique_widths = self.unique_widths_ms/1000
        
        self.unique_fluences = np.array(
            [
                0.5 * (fluence_bins[i] + fluence_bins[i + 1])
                for i in range(len(fluence_bins) - 1)
            ]
        )
        self.unique_peak_fluxes = np.array(
            [
                0.5 * (peak_flux_bins[i] + peak_flux_bins[i + 1])
                for i in range(len(peak_flux_bins) - 1)
            ]
        )

    def package_for_lunfit(self):

        self.det_frac_matrix_snr = self.detection_fraction_peak_flux

        #set the parametes that will be used by LuNfit
        self.detected_bin_midpoints_snr = [self.unique_peak_fluxes, self.unique_widths]
        self.detected_det_frac_snr = self.detection_fraction_peak_flux

        self.detected_bin_midpoints_fluence = [self.unique_fluences, self.unique_widths]
        self.detected_det_frac_fluence = self.detection_fraction_fluence
        #plot the detection fraction as a function of fluence and effective width
        plt.figure()
        plt.pcolormesh(
            np.log10(self.detected_bin_midpoints_fluence[0]),
            np.log10(self.detected_bin_midpoints_fluence[1]),
            self.detected_det_frac_fluence.T,
            shading="auto",
        )
        plt.colorbar(label="Detection Fraction")
        plt.xlabel("log Fluence (Jy ms)")
        plt.ylabel("log Effective Width (s)")
        plt.title("Detection Fraction in Fluence vs Effective Width Bins")
        plt.savefig("detection_fraction_fluence_width.png")
        plt.close()

        plt.figure()
        plt.pcolormesh(
            np.log10(self.detected_bin_midpoints_snr[0]),
            np.log10(self.detected_bin_midpoints_snr[1]),
            self.detected_det_frac_snr.T,
            shading="auto",
        )
        plt.colorbar(label="Detection Fraction")
        plt.xlabel("log Peak Flux (Jy)")
        plt.ylabel("log Effective Width (s)")
        plt.title("Detection Fraction in Peak Flux vs Effective Width Bins")
        plt.savefig("detection_fraction_peakflux_width.png")
        plt.close()

        #change this later, this is arbitrary
        self.detect_error_snr = 1
        self.detect_error_width = 1
        self.detect_error_snr_low_width = 1
        self.detect_error_width_low_width = 1
        self.detect_error_fluence = 1
        # save self
        with open("selection.dill", "wb") as of:
            dill.dump(self, of)





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
    plt.savefig("beam_xy_distribution.png")
    plt.close()

    selection = selection_fluence_width(injections_data_obj)
    # selection.test_selection()
    selection.bin_fluence_dm()
    selection.package_for_lunfit()

    # selection.forward_model_amp_det()
    # selection.plot_modelled_selection_effects()
