import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import basinhopping
import dill
def gen_log(x, B, v, M, cutoff):
    y = 1 / ((1 + np.exp(-B * (x - M))) ** (1 / v))
    y[x < cutoff] = 0
    return y


def forward_model(x, k1, k2, x0, cutoff):
    # this is just a wrapper function so that I only need to change this one reference to change the function used
    return gen_log(x, k1, k2, x0, cutoff)
    # return piecewise_tanh(x, k1, k2, x0, cutoff)
    # return piecewise_logistic(x, k1, k2, x0, cutoff)


class forward_model_class:
    def __init__(self):
        pass
    
        
    def forward_model_det(self):
        karr = []
        self.forward_model_cutoffs = []
        for i in range(len(self.unique_widths)):
            snrs = self.unique_snrs
            det_fracs = self.det_frac_matrix_snr[:, i]
            # determine where x0 is
            snr_interp_arr = np.linspace(0, max(snrs), 1000)
            interp_det_fracs = np.interp(snr_interp_arr, snrs, det_fracs)
            # find where it's closest to 0.5
            x0 = snr_interp_arr[np.argmin(np.abs(interp_det_fracs - 0.5))]
            from scipy.stats import norm

            def p_det_st(x, k1, k2, x0, det_err, cutoff):
                sdet = np.linspace(min(x) - 3 * det_err, max(x) + 3 * det_err, 1000)
                sdet_giv_st = norm.pdf(sdet[:, np.newaxis], loc=x, scale=det_err)
                # pdet_giv_sdet = peicewise_logistic(sdet,k1,k2,x0,cutoff)[:,np.newaxis]
                # pdet_giv_sdet = gen_log(sdet,k1,k2,x0,cutoff)[:,np.newaxis]
                pdet_giv_sdet = forward_model(sdet, k1, k2, x0, cutoff)[:, np.newaxis]
                integral = np.trapz(sdet_giv_st * pdet_giv_sdet, sdet, axis=0)
                return integral

            def loglike(X, x0, snr_arr, det_fracs, det_err, cutoff):
                # assume bernoulli errors for 50 trials
                sigma = X[3]
                # scale sigma by det_fracs
                # sigma = sigma*det_fracs+0.001
                # use a gaussian likelihood
                loglike = np.sum(
                    -0.5
                    * (p_det_st(snr_arr, X[0], X[1], X[2], det_err, cutoff) - det_fracs)
                    ** 2
                    / sigma**2
                    - np.log(sigma * np.sqrt(2 * np.pi))
                )
                return -1 * loglike

            cutoff = np.argwhere(det_fracs < 0.05)
            cutoff = np.max(cutoff)
            self.forward_model_cutoffs.append(snrs[cutoff])
            if self.unique_widths[i] > 4e-3:
                args = (x0, snrs, det_fracs, self.detect_error_snr, snrs[cutoff])
            else:
                # print(f"using low width to model the selection effects")
                # args = (x0,snrs,det_fracs,self.detect_error_snr_low_width,snrs[cutoff])
                args = (x0, snrs, det_fracs, self.detect_error_snr, snrs[cutoff])

            bounds = [(0, 50), (0, 50), (0, 20), (0.01, 0.1)]
            minimizer_kwargs = dict(method="Nelder-Mead", args=args, bounds=bounds)
            init = [1, 1, x0, 0.02]

            res = basinhopping(
                loglike, init, minimizer_kwargs=minimizer_kwargs, niter=50
            )
            # fit the model
            arg_closest_width_det = np.argmin(
                np.abs(self.detected_bin_midpoints_snr[1] - self.unique_widths[i])
            )
            k1, k2, x0, sigma = res.x
            print(
                f"fitted sigma {sigma} k1 {k1} k2 {k2} x0 {x0} cutoff {snrs[cutoff]} width {self.unique_widths[i]}"
            )
            karr.append(res.x)
            plt.figure()
            # plt.plot(snr_interp_arr,peicewise_logistic(snr_interp_arr,k1,k2,x0,snrs[cutoff]),label="forward model")
            plt.plot(
                snr_interp_arr,
                forward_model(snr_interp_arr, k1, k2, x0, snrs[cutoff]),
                label="forward model",
            )
            plt.plot(
                snr_interp_arr,
                p_det_st(
                    snr_interp_arr, k1, k2, x0, self.detect_error_snr, snrs[cutoff]
                ),
                label="forward model st_det",
            )
            plt.plot(snrs, det_fracs, "x", label="data inj")
            plt.plot(
                self.detected_bin_midpoints_snr[0],
                self.detected_det_frac_snr[:, arg_closest_width_det],
                label=f"data det width {self.detected_bin_midpoints_snr[1][arg_closest_width_det]}",
            )
            plt.legend()
            plt.savefig(f"width_{self.unique_widths[i]}_foreward_model.png")
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



if __name__ == "__main__":
    test = TestForwardModelAW()
    test.load_data()
    test.compute_detection_fractions()
    test.forward_model_det()
    test.generate_forward_model_grid()
