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
    #invert the gen_log function
    y = K - gen_log(x, B, v, K, M, 0)
    y[x > cutoff] = 0
    return y

def gen_log_2d(mesh_amp, mesh_width, K_amp, K_width, cutoff_amp, cutoff_width):
    #K_amp are the amplitude parameters, and K_width are the width parameters
    # y = gen_log(mesh_amp, K_amp[0], K_amp[1], K_amp[2], K_amp[3], cutoff_amp) * \
        # flipped_gen_log(mesh_width, K_width[0], K_width[1], K_width[2], K_width[3], cutoff_width)
    #make y a product of gen log and a 4th order polynomial
    y = gen_log(mesh_amp, K_amp[0], K_amp[1], K_amp[2], K_amp[3], cutoff_amp) * \
        np.polyval([K_width[0], K_width[1], K_width[2], K_width[3]], mesh_width)
    y[y<0] = 0
    y[y>1] = 1
    return y

    





def forward_model(x, k1, k2, k3, x0, cutoff):
    # this is just a wrapper function so that I only need to change this one reference to change the function used
    return gen_log(x, k1, k2, k3, x0, cutoff)
    # return piecewise_tanh(x, k1, k2, x0, cutoff)
    # return piecewise_logistic(x, k1, k2, x0, cutoff)

def get_formed_beam():
    from beam_model import formed
    formed_beam_model = formed.FFTFormedActualBeamModel()
    beam_id_base = np.arange(0,256)
    freqs = np.array([600])
    beam_x = []
    beam_y = []
    for i in range(4):
        beam_ids = beam_id_base + (i*1000)
        beam_positions = formed_beam_model.get_beam_positions(beam_ids, freqs)
        for pos in beam_positions:
            beam_x.append(pos[0][0])
            beam_y.append(pos[0][1])

    return np.min(beam_x), np.max(beam_x), np.min(beam_y), np.max(beam_y)

#inherit from injectionsData to add processed attributes
class injectionsData_processed(injectionsData):
    def __init__(self, injection_dict, detection_dict=None):
        super().__init__(injection_dict, detection_dict)

    #overload with a second method where you can just import an injectionsData object
    @classmethod
    def from_injectionsData(cls, inj_data_obj):
        return cls(inj_data_obj.injection, inj_data_obj.detection)

    def process_injections_data(self):
        self.injection_dm = self.injection.get('dm', None)
        self.injection_fluence_jy_ms = self.injection.get('fluence_jy_ms', None)
        self.injection_pulse_width_ms = self.injection.get('pulse_width_ms', None)
        self.injection_tau_1_ghz_ms = self.injection.get('extra_injection_parameters', {}).get('tau_1_ghz_ms', None)
        self.beam_x = self.injection.get('extra_injection_parameters', {}).get('beam_x', None)
        self.beam_y = self.injection.get('extra_injection_parameters', {}).get('beam_y', None)

        self.detected = False

        if self.detection is not None:
            self.detection_snr = self.detection.get('combined_snr', None)
            self.detection_dm = self.detection.get('dm', None)
            self.detected = True
        else:
            self.detection_snr = None
            self.detection_dm = None

        #delete the injection and detection dicts to save memory and storage
        del self.injection
        del self.detection

class selection_fluence_width():
    def __init__(self, injections_data_arr):
        self.injections_data_arr = injections_data_arr
        self.load_tau_width_to_effective_width_map()

    def load_tau_width_to_effective_width_map(self,npz_file='effective_widths.npz'):
        data = np.load(npz_file)
        taus = data['taus']
        sigmas = data['sigmas']
        #first axis is sigma, second axis is tau
        self.mtaus = data['mtaus']
        self.msigmas = data['msigmas']
        self.effective_widths = data['effective_widths']

    def interpolate_effective_width(self, tau, sigma):
        interpolator = interp.RegularGridInterpolator((self.mtaus[0,:], self.msigmas[:,0]), self.effective_widths.T, bounds_error=False, fill_value=None)
        point = np.array([[tau, sigma]])
        effective_width = interpolator(point)[0]
        return effective_width


    def test_selection(self):
        injection_dm = np.array([obj.injection_dm for obj in self.injections_data_arr])
        detection_dm = np.array([obj.detection_dm for obj in self.injections_data_arr])
        plt.figure()
        plt.scatter(injection_dm, detection_dm, c='blue', alpha=0.5)
        plt.xlabel('Injection DM')
        plt.ylabel('Detection DM')
        plt.title('Injection DM vs Detection DM')
        plt.show()


    def bin_fluence_dm(self):
        fluences = np.array([obj.injection_fluence_jy_ms for obj in self.injections_data_arr])
        pulse_width_ms = np.array([obj.injection_pulse_width_ms for obj in self.injections_data_arr])
        tau_1_ghz_ms = np.array([obj.injection_tau_1_ghz_ms for obj in self.injections_data_arr])
        #convert this to scattering timescale at 600mhz
        tau_600_mhz_ms = tau_1_ghz_ms * (1000/600)**4
        #calculate the effective width by interpolating the npzfile
        effective_width = np.array([self.interpolate_effective_width(tau, pw) for tau, pw in zip(tau_600_mhz_ms, pulse_width_ms)])
        


        print("min max tau 600 mhz", np.min(tau_600_mhz_ms), np.max(tau_600_mhz_ms))
        print("min max width", np.min(pulse_width_ms), np.max(pulse_width_ms))
        detected = np.array([obj.detected for obj in self.injections_data_arr])
        #make a 2d histogram of fluence vs pulse width, color coded by detection fraction
        fluence_bins = np.logspace(np.log10(np.min(fluences[fluences>0])), np.log10(np.max(fluences)), 10)
        #make 11 bins in width and 10 in fluence so that it's easier to track
        effective_width_bins = np.logspace(np.log10(1), np.log10(50), 11)
        pulse_width_bins = np.logspace(np.log10(1), np.log10(50), 11)
        tau_600_mhz_ms_bins = np.logspace(np.log10(1), np.log10(50), 11)
        #plot a 2d histogram of fluence vs pulse width, color coded by effective width
        effective_width_av = np.zeros((len(tau_600_mhz_ms_bins)-1, len(pulse_width_bins)-1))
        for i in range(len(tau_600_mhz_ms_bins)-1):
            for j in range(len(pulse_width_bins)-1):
                in_bin = (tau_600_mhz_ms >= tau_600_mhz_ms_bins[i]) & (tau_600_mhz_ms < tau_600_mhz_ms_bins[i+1]) & \
                            (pulse_width_ms >= pulse_width_bins[j]) & (pulse_width_ms < pulse_width_bins[j+1])
                if np.sum(in_bin) > 0:
                    #average all the in_bin effective widths
                    effective_width_av[i, j] = np.mean(effective_width[in_bin])
                else:
                    effective_width_av[i, j] = np.nan
        # These all load the default best-fit model (see model-selection.ipynb)
        # 
        detection_fraction = np.zeros((len(fluence_bins)-1, len(pulse_width_bins)-1))
        for i in range(len(fluence_bins)-1):
            for j in range(len(effective_width_bins)-1):
                in_bin = (fluences >= fluence_bins[i]) & (fluences < fluence_bins[i+1]) & \
                            (effective_width >= effective_width_bins[j]) & (effective_width < effective_width_bins[j+1])
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
        #set to midpoints of the bin edges
        self.unique_widths = np.array([0.5 * (effective_width_bins[i] + effective_width_bins[i+1]) for i in range(len(effective_width_bins)-1)])
        # self.fluence_bin_edges = fluence_bin_edges
        #set this as unique snrs too
        self.unique_snrs = np.array([0.5 * (fluence_bins[i] + fluence_bins[i+1]) for i in range(len(fluence_bins)-1)])
        self.unique_amplitude = self.unique_snrs

        #save self
        with open("temp.dill", "wb") as of:
            dill.dump(self, of)

       

        plt.figure()
        plt.pcolormesh(tau_600_mhz_ms_bins, pulse_width_bins, effective_width_av.T, shading='auto', cmap='viridis')
        plt.xlabel('Tau 1 GHz (ms)')
        plt.ylabel('Pulse Width (ms)')
        plt.colorbar(label='Average Effective Width (ms)')
        plt.xscale('log')
        plt.yscale('log')
        plt.savefig('effective_width_vs_tau_width.png')

        plt.figure()
        plt.pcolormesh(fluence_bins, effective_width_bins, detection_fraction.T, shading='auto', cmap='viridis')
        plt.xlabel('Fluence (Jy ms)')
        plt.ylabel('Pulse Width (ms)')
        plt.colorbar(label='Detection Fraction')
        plt.xscale('log')
        plt.yscale('log')
        plt.title('Detection Fraction vs Fluence and Pulse Width')
        plt.savefig('detection_fraction_vs_fluence_width.png')
        #plot some slices of the selection function in fluence at fixed pulse widths
        plt.figure(figsize=(10, 8))
        fluence_axis = 0.5 * (fluence_bins[:-1] + fluence_bins[1:])
        i = 0
        while i<len(effective_width_bins)-1:
            #do every 10th pulse width bin
            mid_pulse_width = 0.5 * (effective_width_bins[i] + effective_width_bins[i+1])
            plt.plot(fluence_axis, detection_fraction[:, i], label=f'{mid_pulse_width}')
            i += 1
            # ax[j].set_xscale('log')
            # ax[j].set_yscale('log')
            # set the x axis from 0-50
        plt.xscale('log')
        plt.ylabel('Selection Probability')
        plt.xlabel('Fluence (Jy ms)')
        plt.legend()
        plt.savefig('selection_function_slices_fluence.png')

    def forward_model_wdith_amp(self):
        #this function will forward model both width and amplitude such that you can get pdet|fluence,width
        unique_amps = self.unique_amplitude
        unique_widths = self.unique_widths
        detection_fraction = self.detection_fraction

        def p_det_st_wt(u_amps, u_widths, K_amp, K_width, width_err, amp_err, cutoff_amp, cutoff_width):
            # #create a 2d array of w_det and s_det
            # min_sdet = min(u_amps)-3*amp_err
            # if min_sdet < 0:
            #     min_set = 0
            # sdet = np.linspace(min_sdet, max(u_amps)+3*amp_err, 100)
            # min_width = min(u_widths)-3*width_err
            # if min_width < 0:
            #     min_width = 0
            
            # wdet = np.linspace(min_width, max(u_widths)+3*width_err, 101)
            # sdet_mesh, wdet_mesh = np.meshgrid(sdet, wdet)
            # pdet_giv_sdet_wdet = gen_log_2d(sdet_mesh, wdet_mesh, K_amp, K_width, cutoff_amp, cutoff_width)
            # # create a gaussian distribution in the amp and width directions, assume independence, so we can do the two 1d gaussians independently and multiply
            # amp_gauss = norm.pdf(sdet[:,np.newaxis], loc=u_amps[np.newaxis,:], scale=amp_err)  # shape (len(sdet), len(u_amps), in the future, amp_err can be an array too)
            # width_gauss = norm.pdf(wdet[:,np.newaxis], loc=u_widths[np.newaxis,:], scale=width_err)  # shape (len(wdet), len(u_widths))

            # pdet_giv_wdet = pdet_giv_sdet_wdet[:,:,np.newaxis]  * amp_gauss[np.newaxis,:,:] 
            # #integrate over sdet
            # pdet_giv_wdet = np.trapezoid(pdet_giv_wdet, sdet, axis=1)  # shape (len(wdet), len(u_amps))
            # pdet = pdet_giv_wdet[:,:, np.newaxis] * width_gauss[:,np.newaxis,:]  # shape (len(u_amps), len(u_widths))
            # #integrate over wdet
            # pdet = np.trapezoid(pdet, wdet, axis=0)  # shape (len(u_amps), len(u_widths))
             
            sdet_mesh, wdet_mesh = np.meshgrid(u_amps, u_widths)
            pdet = gen_log_2d(sdet_mesh, wdet_mesh, K_amp, K_width, cutoff_amp, cutoff_width).T
            # plt.figure()
            # plt.pcolormesh(np.log10(u_widths), np.log10(u_amps), pdet, shading='auto', cmap='viridis')
            # plt.xlabel('Width (ms)')
            # plt.ylabel('Amplitude')
            # plt.colorbar(label='Detection Fraction before smearing')
            # plt.figure()
            # plt.pcolormesh(np.log10(unique_widths), np.log10(unique_amps), detection_fraction, shading='auto', cmap='viridis')
            # #change to log log
            # plt.xlabel('Width (ms)')
            # plt.ylabel('Amplitude')
            # plt.colorbar(label='Detection Fraction')

            # plt.show()
            return pdet

        def loglike_2d(X, u_amps, u_widths, det_fracs, amp_err, width_err, cutoff_amp, cutoff_width):
            sigma = X[-1]
            loglike = -0.5*np.nansum(p_det_st_wt(u_amps, u_widths, X[0:4], X[4:8], amp_err, width_err, cutoff_amp, cutoff_width) - det_fracs)**2 / sigma**2
            loglike -= np.log(sigma * np.sqrt(2 * np.pi))
            return -1 * loglike

        amplitude_cutoff = 0
        width_cutoff = 100
        x0 = 15
        w0 = 60
        init = [0.1, 10, 1, x0, 1, 10, 1, w0, 1]
        # bounds = [(0, 50), (0, 50), (0,1), (0, 50),(0, 50), (0, 50), (0,1), (0, 100), (0.01, 1)]
        bounds = [(0, 50), (0, 50), (0,10), (0, 50),(-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), (0.01, 1)]
        amp_err = 1
        width_err = 1
        minimizer_kwargs = dict(method="Nelder-Mead", args=(unique_amps, unique_widths, detection_fraction, amp_err, width_err, amplitude_cutoff, width_cutoff), bounds=bounds)
        res = basinhopping(
            loglike_2d, init, minimizer_kwargs=minimizer_kwargs, niter=1000
        )
        plot = True
        #print the loglikes
        print(loglike_2d(res.x, unique_amps, unique_widths, detection_fraction, amp_err, width_err, amplitude_cutoff, width_cutoff))
        print(loglike_2d(init, unique_amps, unique_widths, detection_fraction, amp_err, width_err, amplitude_cutoff, width_cutoff))
        if plot:
            plt.figure()
            plt.pcolormesh(np.log10(unique_widths), np.log10(unique_amps), detection_fraction, shading='auto', cmap='viridis')
            #change to log log
            plt.xlabel('Width (ms)')
            plt.ylabel('Amplitude')
            plt.colorbar(label='Detection Fraction')
            #set colorbar lim to 0-1
            plt.clim(0, 1)
            plt.figure()
            plt.pcolormesh(np.log10(unique_widths), np.log10(unique_amps), p_det_st_wt(unique_amps, unique_widths, res.x[0:4], res.x[4:8], amp_err, width_err, amplitude_cutoff, width_cutoff), shading='auto', cmap='viridis')
            plt.xlabel('Width (ms)')
            plt.ylabel('Amplitude')
            plt.colorbar(label='Forward Modelled Detection Fraction')
            plt.clim(0, 1)
            plt.show()


        import pdb; pdb.set_trace()

        
        





    def forward_model_det(self):
        karr = []
        self.forward_model_cutoffs = []
        for i in range(len(self.unique_widths)):
            snrs = self.unique_snrs

            snrs = np.array(snrs)
            det_fracs = self.det_frac_matrix_snr[:, i]
            det_fracs = np.array(det_fracs)
            snr_mask  = snrs<1000
            det_fracs = det_fracs[snr_mask]
            snrs = snrs[snr_mask]
            # determine where x0 is
            snr_interp_arr = np.linspace(0, max(snrs), 1000)

            interp_det_fracs = np.interp(snr_interp_arr, snrs, det_fracs)
            # find where it's closest to 0.5
            x0 = snr_interp_arr[np.argmin(np.abs(interp_det_fracs - 0.5))]

            def p_det_st(x, k1, k2, k3, x0, det_err, cutoff):
                sdet = np.linspace(min(x) - 3 * det_err, max(x) + 3 * det_err, 1000)
                sdet_giv_st = norm.pdf(sdet, loc=x, scale=det_err)
                # pdet_giv_sdet = peicewise_logistic(sdet,k1,k2,x0,cutoff)[:,np.newaxis]
                # pdet_giv_sdet = gen_log(sdet,k1,k2,x0,cutoff)[:,np.newaxis]
                pdet_giv_sdet = forward_model(sdet, k1, k2,k3, x0, cutoff)
                integral = np.trapz(sdet_giv_st * pdet_giv_sdet, sdet, axis=0)
                return integral

            def loglike(X, snr_arr, det_fracs, det_err, cutoff):
                sigma = 1
                # scale sigma by det_fracs
                # use a gaussian likelihood
                loglike = np.nansum(
                    -0.5
                    * (p_det_st(snr_arr, X[0], X[1], X[2], X[3], det_err, cutoff) - det_fracs)
                    ** 2
                    / sigma**2
                    - np.log(sigma * np.sqrt(2 * np.pi))
                )
                return -1 * loglike

            cutoff = np.argwhere(det_fracs < 0.05)
            cutoff = np.max(cutoff)
            self.forward_model_cutoffs.append(snrs[cutoff])
            self.detect_error_snr = np.linspace(0.5, 2.0, len(snrs))
            bounds = [(0, 50), (0, 50),(0,1), (0, 20), (0.01, 0.1)]
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
            # plt.plot(snr_interp_arr,peicewise_logistic(snr_interp_arr,k1,k2,x0,snrs[cutoff]),label="forward model")
            plt.plot(
                snr_interp_arr,
                forward_model(snr_interp_arr, k1, k2, k3, x0, snrs[cutoff]),
                label="forward model",
            )
            plt.plot(
                snrs,
                p_det_st(
                    snrs, k1, k2, k3, x0, self.detect_error_snr, snrs[cutoff]
                ),
                label="forward model pdet | st",
            )

            plt.plot(snrs, det_fracs, "x", label="data inj")
            plt.legend()
            plt.savefig(f"width_{self.unique_widths[i]}_foreward_model.png")
            plt.show()
            # plt.show()
            plt.close()
        self.karr = karr
        with open("test.dill", "wb") as of:
            dill.dump(self, of)

    def generate_forward_model_grid(
        self,
    ):
        self.forward_model_snr_arrs = np.linspace(0, 50, 1000)
        self.det_frac_foreward_model_matrix_snr = np.zeros(
            (len(self.forward_model_snr_arrs), len(self.unique_widths))
        )
        for i in range(len(self.unique_widths)):
            k1, k2, x0, sigma = self.karr[i]
            print(
                k1, k2, x0, sigma, self.unique_widths[i], self.forward_model_cutoffs[i]
            )
            self.det_frac_foreward_model_matrix_snr[:, i] = forward_model(
                self.forward_model_snr_arrs, k1, k2, x0, self.forward_model_cutoffs[i]
            )
        plt.figure()
        plt.title("forward modelled pdet|sdet")
        plt.pcolormesh(
            self.unique_widths * 1000,
            self.forward_model_snr_arrs,
            self.det_frac_foreward_model_matrix_snr,
        )
        plt.xlabel("width (ms)")
        plt.ylabel("snr")
        plt.savefig("forward_modelled_pdet_sdet.png")
        plt.close()



    # plt.figure()
    # plt.scatter(beam_x, beam_y, alpha=0.5)
    # plt.xlabel('Beam X')
    # plt.ylabel('Beam Y')
    # plt.title('Formed Beam Positions at 600 MHz')
    # plt.show()

    # import pdb; pdb.set_trace()




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process injections data.")
    parser.add_argument('input_file', type=str, help='Path to the input .npy file containing injections data array.')
    args = parser.parse_args()
    beam_x_min, beam_x_max, beam_y_min, beam_y_max = get_formed_beam()

    if args.input_file.endswith('.dill'):
        with open(args.input_file, 'rb') as f:
            selection = dill.load(f)
    else:
        injections_data_arr = np.load(args.input_file, allow_pickle=True)

        injections_data_obj = [injectionsData_processed.from_injectionsData(inj) for inj in injections_data_arr]
        for inj_obj in injections_data_obj:
            inj_obj.process_injections_data()
        beam_x_arr = np.array([inj.beam_x for inj in injections_data_obj])
        beam_y_arr = np.array([inj.beam_y for inj in injections_data_obj])
        #only keep those in the formed beam area
        in_beam = (beam_x_arr >= beam_x_min) & (beam_x_arr <= beam_x_max) & \
                    (beam_y_arr >= beam_y_min) & (beam_y_arr <= beam_y_max)
        injections_data_obj = [inj for i, inj in enumerate(injections_data_obj) if in_beam[i]]
        beam_x_arr = beam_x_arr[in_beam]
        beam_y_arr = beam_y_arr[in_beam]

        plt.figure()
        plt.scatter(beam_x_arr, beam_y_arr, alpha=0.5)
        plt.xlabel('Beam X')
        plt.ylabel('Beam Y')
        plt.show()

        selection = selection_fluence_width(injections_data_obj)
        # selection.test_selection()
        selection.bin_fluence_dm()

    selection.forward_model_det()

