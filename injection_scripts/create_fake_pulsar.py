#!/usr/bin/env python3

from inject_pulses_sigpyproc import multiprocess
import argparse
from simulate_pulse import simulate_pulses
import numpy as np
import matplotlib.pyplot as plt

def create_lognorm_pulses(mu,std,p,f,d,dm,downsamp=3,stats_window=0.9):
    #always start 5 seconds in and end 5 seconds early
    d = d-10

    pulse_snrs = simulate_pulses(obs_t=d,
                                 period=p,
                                 f=f,
                                 mu=mu,
                                 std=std)
    toa = []
    while len(toa)!=len(pulse_snrs):
        rand = np.random.rand(int(d*f/p))
        toa = np.array(list(range(int(d*f/p))))
        toa = toa[rand<f]
        #starting 5 seconds into obs
        toa = (toa*p)+5

    grid_coords = []
    for t,s in zip(toa,pulse_snrs):
        grid_coords.append((t,s,dm))
    grid_coords = np.array(grid_coords)
    np.savez("sample_injections", grid=grid_coords,downsamp=downsamp,stats_window=stats_window)
    return grid_coords,stats_window,downsamp


if __name__ =="__main__":
    #dm,s,ifn,duration,maskfn,injection_sample,stats_window = arr
    parser = argparse.ArgumentParser()
    parser.add_argument("fil", help="Filterbank file to use as raw data background")
    parser.add_argument("-d", help="Duration of required output file", type=float, default=300)
    parser.add_argument("-p", help="Pulse period", type=float, default=2)
    parser.add_argument("-f", help="Pulse fraction", type=float, default=1)
    parser.add_argument("-m", help="this is the mask fn")

    args = parser.parse_args()

    DM = 100
    mu = 0.5
    std = 0.2

    FN_ext = 1234
    ifn = args.fil
    p = args.p
    f = args.f
    d = args.d
    maskfn = args.m
    pulse_attributes,stats_window,downsamp = create_lognorm_pulses(mu=mu,std=std,p=p,f=f,d=d,dm=DM)
    inject_params = (DM,FN_ext,ifn,d,maskfn,pulse_attributes,stats_window,downsamp,False)
    multiprocess(inject_params)
