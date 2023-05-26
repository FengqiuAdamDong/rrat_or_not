#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from statistics import lognorm_dist
from scipy.stats import norm
from multiprocessing import Pool
import gp_emulator

# from statistics import p_detect

# first we need to generate the data to train on
def integral(input_params):
    # x is the input, i.e. snr_det the rest are parameters of either the log normal or the detection error
    if len(input_params) > 1:
        import pdb

        pdb.set_trace()
    mu, std, sigma_snr, mu_snr_arr = input_params[0]
    snr_arr = np.linspace(1e-20, 10, 500)
    snr_m, mu_snr_m = np.meshgrid(snr_arr, mu_snr_arr)
    # take log of the snr distribution
    p_snr = lognorm_dist(snr_m, mu, std)
    p_musnr_giv_snr = norm.pdf(mu_snr_m - snr_m, 0, sigma_snr)
    # combine the two terms
    p_second_conv = p_musnr_giv_snr * p_snr
    p_second_int = np.trapz(p_second_conv, snr_arr)
    if np.isnan(p_second_int):
        import pdb

        pdb.set_trace()
    return p_second_int


if __name__ == "__main__":
    # need to calculate for a range of mus, stds and sigma_snrs
    mesh_size = 10
    mu_arr = np.linspace(1e-20, 10, mesh_size)
    std_arr = np.linspace(1e-20, 10, mesh_size + 1)
    sigma_snr = np.linspace(1e-20, 1, mesh_size + 2)
    snr_det_array = np.linspace(-10, 25, 1000)
    # inputs_array = np.zeros(len(snr_det_array),3)
    # integral_value = np.zeros((len(snr_det_array),len(mu_arr),len(std_arr),len(sigma_snr)))
    # for i,mu in enumerate(mu_arr):
    # for j,std in enumerate(std_arr):
    # for k,sigma in enumerate(sigma_snr):
    # integral_value[:,i,j,k] = integral(snr_det_array,mu,std,sigma)
    parameters = ["mu", "std", "sigma_snr", "snr_det"]
    min_vals = [1e-20, 0.1, 1e-20, -10]
    max_vals = [5, 2, 1, 40]
    n_train = 10000
    n_validate = 300
    x = gp_emulator.create_emulator_validation(
        integral,
        parameters,
        min_vals,
        max_vals,
        n_train,
        n_validate,
        do_gradient=True,
        n_tries=50,
        n_procs=30,
    )
    gp_obj = x[0]
    val_inp = x[1]
    val_out = x[2]
    emulate_val = x[3]
    emulator_predict, emulat_err, _ = gp_obj.predict(val_inp)
    plt.plot(emulator_predict - val_out[:, 0])
    plt.show()
    import pdb

    pdb.set_trace()
