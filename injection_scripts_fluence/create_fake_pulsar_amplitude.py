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
def create_lognorm_pulses(mu,std,p,f,d,dm,downsamp=3,stats_window=0.9):
    #always start 5 seconds in and end 5 seconds early
    d = d-10
    width = 12.41e-3
    print("simulating pulses")
    pulse_amps = simulate_pulses(obs_t=d,
                                 period=p,
                                 f=f,
                                 mu=mu,
                                 std=std)

    toa = np.linspace(0,len(pulse_amps),len(pulse_amps)) * p
    toa = toa-(max(toa)/2)
    toa = toa+(d/2)
    print(toa)

    #need to convert pulse amps to fluence
    pulse_fluence = pulse_amps*width*np.sqrt(2*np.pi)
    grid_coords = []
    for t,s in zip(toa,pulse_fluence):
        grid_coords.append((t,s,dm,width))
    grid_coords = np.array(grid_coords)
    np.savez(
        "sample_injections",
        grid=grid_coords,
        downsamp=downsamp,
        stats_window=stats_window,
    )

    print(len(pulse_amps))
    plt.hist(pulse_amps,bins="auto",density=True)
    x = np.linspace(0.001,1,10000)
    plt.plot(x,lognorm_dist(x,mu,std))
    plt.figure()
    plt.hist(pulse_fluence,bins="auto",density=True)
    plt.show()
    return grid_coords,stats_window,downsamp


if __name__ =="__main__":
    #dm,s,ifn,duration,maskfn,injection_sample,stats_window = arr
    parser = argparse.ArgumentParser()
    parser.add_argument("fil", help="Filterbank file to use as raw data background")
    parser.add_argument("-d", help="Duration of required output file", type=float, default=700)
    parser.add_argument("-p", help="Pulse period", type=float, default=2)
    parser.add_argument("-f", help="Pulse fraction", type=float, default=0.8)
    parser.add_argument("-m", help="this is the mask fn")

    args = parser.parse_args()

    DM = 100
    mu = -3
    std = 0.2
    FN_ext = 2345
    ifn = args.fil
    p = args.p
    f = args.f
    d = args.d
    maskfn = args.m
    pulse_attributes,stats_window,downsamp = create_lognorm_pulses(mu=mu,std=std,p=p,f=f,d=d,dm=DM)
    inject_params = (DM,FN_ext,ifn,d,maskfn,pulse_attributes,stats_window,downsamp,False)
    multiprocess(inject_params)
