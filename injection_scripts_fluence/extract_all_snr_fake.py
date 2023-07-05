#!/usr/bin/env python3
import argparse
import numpy as np
from inject_stats import inject_obj
from inject_stats import get_mask_fn
from pathos.pools import ProcessPool
import dill
import sys
import csv
import matplotlib.pyplot as plt
#this script will pretend to run a fake pulse through the selection function

class det_obj(inject_obj):
    # det obj is inherited from injection object....
    pass



def combine_positives(fil1_, fil2_, dm1_, dm2_, toa1_, toa2_):
    # this function combines two sets (from positive_bursts_1 and positive_bursts_short eg)
    # fil 1 is the one I'm keeping
    fil_add = []
    dm_add = []
    toa_add = []
    no_match = False
    for fil2, dm2, toa2 in zip(fil2_, dm2_, toa2_):
        for fil1, dm1, toa1 in zip(fil1_, dm1_, toa1_):
            no_match = False
            if (fil1 == fil2) & (dm1 == dm2) & (toa1 == toa2):
                break
            no_match = True
        if no_match:
            fil_add.append(fil2)
            dm_add.append(dm2)
            toa_add.append(toa2)
    return np.append(fil1_, fil_add), np.append(dm1_, dm_add), np.append(toa1_, toa_add)


if __name__ == "__main__":
    # fn = 'real_pulses/positive_bursts_edit_snr.csv'
    # fn1 = 'real_pulses/positive_bursts_1_edit_snr.csv'
    # fn2 = 'real_pulses/positive_bursts_short_edit_snr.csv'
    # fn = 'real_pulses/positive_burst_test.csv'
    # fn1 = fn
    # fn2 = fn
    # be sure to cutout all of the filterbank files first!!
    # dedisperse to some nominal DM to make it easier
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-dm",
        default=0,
        help="dm to dedisperse to, if not given, will use positive burst file",
        required=True
    )
    parser.add_argument("-p", nargs="+", help="TOA files")
    parser.add_argument(
        "-ds", type=int, help="The downsample when getting det_snr", required=True
    )
    parser.add_argument(
        "-fil", type=str, help="Filterbank file", required=True
    )

    args = parser.parse_args()

    dm = float(args.dm)
    # in the cut out the pulse is always at 3s
    positive_fl = args.p
    downsamp = args.ds
    fil = args.fil
    fake_pulsar_params = np.load("sample_injections.npz")['grid']
    toas = fake_pulsar_params[:,0]
    dm = fake_pulsar_params[:,2]
    width = fake_pulsar_params[:,3]
    snr = fake_pulsar_params[:,1]

    #create an array that is
    maskfn = fil.strip(".fil") + "_rfifind.mask"

    detection_obj = det_obj(snr = snr,width=width,toas=toas,
                            dm=dm,downsamp=downsamp,filfile=fil,mask=maskfn)
    detection_obj.calculate_fluence()
    #load the detection fn
    #import dill
    from simulate_pulse import n_detect
    pulses = detection_obj.det_snr


    with open("inj_stats_fitted.dill", "rb") as inf:
        inj_stats = dill.load(inf)
    poly_params = inj_stats.poly_snr
    poly_fun = np.poly1d(poly_params)
    pulses_altered = np.array(poly_fun(pulses))

    simulated_det_pulses = n_detect(pulses)
    simulated_det_pulses_altered = n_detect(pulses_altered)
    nd = []
    nd_altered = []
    for i in range(10000):
        nd.append(len(n_detect(pulses)))
        nd_altered.append(len(n_detect(pulses_altered)))


    print("non-altered",len(simulated_det_pulses))
    print("altered",len(simulated_det_pulses_altered))
    fig,axes = plt.subplots(1,3)
    axes[0].hist(snr,bins="auto",label="injected",alpha=0.5)
    axes[0].hist(pulses,bins="auto",label="all",alpha=0.5)
    axes[0].hist(simulated_det_pulses,bins="auto",alpha=0.5,label="detected")
    axes[0].legend()
    axes[1].scatter(snr,pulses)
    axes[1].set_xlabel("injected")
    axes[1].set_ylabel("retrieved")
    axes[2].hist(nd,alpha=0.5,bins="auto",label="non altered")
    axes[2].hist(nd_altered,alpha=0.5,bins="auto",label="non altered")
    axes[2].legend()
    plt.show()

    import dill                            #pip install dill --user
    filename = 'globalsave.pkl'
    dill.dump_session(filename)

    import pdb; pdb.set_trace()
