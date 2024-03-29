#!/usr/bin/env python3

import dill
import numpy as np
import sys
from simulate_pulse import simulate_pulses
from simulate_pulse import n_detect
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
def simulate_and_process_data(detected_req, mode,mu_ln, std_ln, lower, upper, sigma_snr, a, inj_file, dill_file,plot=True,out_fol="simulated_dir"):
    detected_pulses = []
    total_pulses = []
    while len(detected_pulses) < detected_req:
        obs_t = 1
        p = 1
        f = 1
        if mode == "Exp":
            pulses = simulate_pulses_exp(obs_t, p, f, mu_ln, a, random=False,lower=lower,upper=upper)
        elif mode == "Lognorm":
            pulses = simulate_pulses(
                obs_t, p, f, mu_ln, std_ln, a, lower=lower, upper=upper, random=False
            )
        elif mode == "Gauss":
            pulses, a, b = simulate_pulses_gauss(obs_t, p, f, mu_ln, std_ln, random=True)
        rv = norm(loc=0, scale=sigma_snr).rvs(len(pulses))
        d_pulses = rv + pulses
        d = n_detect(d_pulses, inj_file)
        if len(d) == 1:
            detected_pulses.append(d[0])
        total_pulses.append(pulses)

    print("len detected", len(detected_pulses))
    print("generated", len(total_pulses))
    print("mean", np.mean(total_pulses), "variance", np.std(total_pulses) ** 2)
    print("detection error", sigma_snr)
    print("detected", len(detected_pulses))
    out_fn = os.path.join(out_fol,f"simulated_{mode}_{detected_req}_{mu}_{std}_{a}.dill")
    if not os.path.exists(out_fol):
        os.makedirs(out_fol)
    detected_pulses = process_data(dill_file, detected_pulses, out_fn, plot=True)
    yaml_fn = os.path.join(out_fol,f"simulated_{mode}_{detected_req}_{mu}_{std}_{a}.yaml")
    if mode == "Lognorm":
        write_yaml(mu_ln,std_ln,a,len(total_pulses),inj_file,yaml_fn)
    elif mode == "Exp":
        write_yaml_exp(mu_ln,a,len(total_pulses),inj_file,yaml_fn)
    if plot:
        snr_array = np.linspace(0, 20, 10000)

        plt.figure()
        if mode == "Lognorm":
            p_dist, p_det, conv_amp_array, conv = statistics.first_plot(
                snr_array, mu_ln, std_ln, sigma_snr, a=a, lower_c=lower, upper_c=upper
            )

            plt.plot(snr_array+a, lognorm_dist(snr_array, mu_ln, std_ln), label="lognorm")
        elif mode == "Exp":
            import statistics_exp
            print(statistics_exp.det_error)
            p_dist = statistics_exp.first_exp_plot(snr_array, mu_ln, sigma_snr)
            plt.plot(snr_array, expon.pdf(snr_array, scale=mu_ln), label="expon")
        p_dist = p_dist / np.trapz(p_dist, snr_array)
        plt.hist(detected_pulses, bins="auto", density=True, alpha=0.5, label="new fake data")
        plt.plot(snr_array, statistics.p_detect(snr_array), label="det_prob")
        plt.plot(snr_array, p_dist, label="detection function")
        plt.legend()
        plt.show()

def write_yaml(mu,std,a,N,inj_file,output_fn):
    mu = float(mu)
    mu_arr = [mu-1,mu+1]
    data = {
        'logn_N_range': [-1,(N)*2],
        'logn_mu_range': list(mu_arr),
        'logn_std_range': [std-0.5, std+0.5],
        'exp_N_range': [-1,N*2],
        'exp_k_range': [0.1, 10],
        'detection_curve': inj_file,
        'snr_thresh': 1.3,
        'a': a,
        'N': N,
        'mu': mu,
        'std': std,
    }
    with open(output_fn,'w') as my_file:
        yaml.dump(data, my_file)

def write_yaml_exp(k,a,N,inj_file,output_fn):
    #in this case mu is the k variable for the exponential distribution
    k = float(k)
    k_low = k-1
    if k_low < 0:
        k_low = 0.1
    k_arr = [k_low,k+1]

    data = {
        'exp_N_range': [-1,N*2],
        'exp_k_range': list(k_arr),
        'logn_N_range': [-1,N*2],
        'logn_mu_range': [-2,2],
        'logn_std_range': [0.01, 2],
        'detection_curve': inj_file,
        'snr_thresh': 1.3,
        'a': a,
        'N': N,
        'k': k,
    }
    with open(output_fn,'w') as my_file:
        yaml.dump(data, my_file)



def process_data(dill_file,detected_data,output_fn, plot=True):
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

    det_snr = []
    for pulse_obj in det_class.sorted_pulses:
        if pulse_obj.det_snr != -1:
            det_snr.append(pulse_obj.det_snr)

    if plot:
        plt.figure()
        plt.hist(det_snr, bins="auto", density=True, label="real data")
        plt.hist(detected_data, bins="auto", density=True, label="fake data",alpha=0.5)
        plt.legend()

    filfiles = np.full(len(det_snr), "abc", dtype=str)
    maskfn = np.full(len(det_snr), "abc", dtype=str)
    dms = np.full(len(det_snr), 123, dtype=float)
    toas = np.full(len(det_snr), 123, dtype=float)

    inject_obj_arr = []
    for snr in detected_data:
        #create a new inject_obj and just fill with fake data
        temp = inject_obj()
        temp.det_snr = snr
        temp.det_fluence = snr
        temp.det_amp = snr
        temp.fluence_amp = snr
        temp.det_std = snr
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

    det_snr = []
    for pulse_obj in det_class.sorted_pulses:
        if pulse_obj.det_snr != -1:
            det_snr.append(pulse_obj.det_snr)
    return det_snr



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
parser.add_argument("-of", type=str, default="simulated_dir",required=True)
parser.add_argument("-detected_req", type=int, default=10000)
args = parser.parse_args()
a = args.a
dill_file = args.d
inj_file = args.inj_file
output_fol = args.of
mode = args.mode
#load the detection file
statistics_basic.load_detection_fn(inj_file,plot=False)
if mode=="Lognorm":
    mu_arr = np.linspace(-0.5,1,20)
    std_arr = [args.s]
elif mode=="Exp":
    k_arr = np.linspace(0.5,2,20)
    #use mu as k_arr
    mu_arr = k_arr
    #the std arr is ignored for exp distribution
    std_arr = [0.5]

detected_req = args.detected_req
print("mu", mu_arr, "std", std_arr, "a", a, "detected_req", detected_req)
from inject_stats import inject_obj
from numpy.random import normal
import statistics
from statistics import lognorm_dist
from statistics import mean_var_to_mu_std
from statistics import mu_std_to_mean_var
from statistics_exp import k_to_mean_var
if __name__ == "__main__":
    sigma_snr = statistics_basic.det_error
    #simulate pulses one at a time
    for mu in mu_arr:
        std = std_arr[0]
        lower = 0
        if mode=="Lognorm":
            #use the upper cutoff to be 50x the mean
            mean,var = mu_std_to_mean_var(mu,std)
            median = np.exp(mu)
            #set upper to 50x median
            upper = 50*median
        elif mode=="Exp":
            mean,var = k_to_mean_var(mu)
            median = np.log(2)/mu
            upper = 50*median
        simulate_and_process_data(detected_req, mode,mu,std, lower, upper, sigma_snr, a, inj_file, dill_file,plot=False,out_fol=output_fol)
