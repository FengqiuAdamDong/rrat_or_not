#!/usr/bin/env python3
from inject_pulses_sigpyproc import multiprocess
import argparse
from simulate_pulse import simulate_pulses
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import normal
def lognorm_dist(x, mu, sigma):
    pdf = np.zeros(x.shape)
    pdf[x > 0] = np.exp(-((np.log(x[x > 0]) - mu) ** 2) / (2 * sigma**2)) / (
        x[x > 0] * sigma * np.sqrt(2 * np.pi)
    )
    return pdf

#this scripts creates a filterbank file that has a fake pulsar in it!!
def create_lognorm_pulses(mu,std,mu_w,std_w,p,f,d,dm,downsamp=3,stats_window=0.9):
    #always start 5 seconds in and end 5 seconds early
    inj_d = d-20
    print("simulating pulses")
    pulse_snrs = simulate_pulses(obs_t=inj_d,
                                 period=p,
                                 f=f,
                                 mu=mu,
                                 std=std,
                                 a=0,
                                 lower=0,
                                 upper=np.inf,
                                 random=False,
                                 )
    pulse_widths = simulate_pulses(obs_t=inj_d,
                                   period=p,
                                   f=f,
                                   mu=mu_w,
                                   std=std_w,
                                   a=0,
                                   lower=0,
                                   upper=np.inf,
                                   random=False,
                                   )
    toa = np.linspace(0,len(pulse_snrs),len(pulse_snrs)) * p
    toa = toa-(max(toa)/2)
    toa = toa+(d/2)

    grid_coords = []
    for t,s,w in zip(toa,pulse_snrs,pulse_widths):
        grid_coords.append((t,s,dm,w))
    grid_coords = np.array(grid_coords)
    np.savez(
        "sample_injections",
        grid=grid_coords,
        downsamp=downsamp,
        stats_window=stats_window,
    )

    plt.hist(pulse_snrs,bins="auto",density=True)
    x = np.linspace(0.001,max(pulse_snrs),10000)
    plt.plot(x,lognorm_dist(x,mu,std))

    plt.figure()
    plt.hist(pulse_widths,bins="auto",density=True)
    x = np.linspace(0.001,max(pulse_widths),10000)
    plt.plot(x,lognorm_dist(x,mu_w,std_w))
    plt.show()

    print(min(toa),max(toa))
    return grid_coords,stats_window,downsamp


if __name__ =="__main__":
    #dm,s,ifn,duration,maskfn,injection_sample,stats_window = arr
    parser = argparse.ArgumentParser()
    parser.add_argument("fil", help="Filterbank file to use as raw data background")
    parser.add_argument("-d", help="Duration of required output file", type=float, default=700)
    parser.add_argument("-p", help="Pulse period", type=float, default=2)
    parser.add_argument("-f", help="Pulse fraction", type=float, default=0.8)
    parser.add_argument("-m", help="this is the mask fn")
    parser.add_argument("-ext", help="this is the extension of the output file, integer", type=int, default=2345)

    args = parser.parse_args()

    DM = 100
    mu = 1.0
    std = 0.2
    mu_w = -6.21
    std_w = 0.2
    FN_ext = float(args.ext)
    ifn = args.fil
    p = args.p
    f = args.f
    d = args.d
    maskfn = args.m
    pulse_attributes,stats_window,downsamp = create_lognorm_pulses(mu=mu,std=std,mu_w=mu_w,std_w=std_w,p=p,f=f,d=d,dm=DM)
    print(f"injecting at dm {DM} with pulse period {p} and pulse fraction {f} into file with duration {d}")
    inject_params = (DM,FN_ext,FN_ext,ifn,d,maskfn,pulse_attributes,stats_window,downsamp,False)
    multiprocess(inject_params)
