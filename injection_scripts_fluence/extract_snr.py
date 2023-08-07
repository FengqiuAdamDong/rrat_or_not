#!/usr/bin/env python3
import argparse
import numpy as np
from inject_stats import inject_obj
from inject_stats import get_mask_fn
from pathos.pools import ProcessPool
import dill
import sys
import csv
import os

class det_obj(inject_obj):
    # det obj is inherited from injection object....
    def __init__(
            self, snr=1, width=1, toas=1, dm=1, downsamp=8, filfile="", mask="", pulse_number=0
    ):
        self.snr = snr
        self.width = width
        self.toas = toas
        self.dm = dm
        self.filfile = filfile
        self.mask = mask
        self.downsamp = downsamp
        self.det_fluence = []
        self.det_amp = []
        self.det_std = []
        self.fluence_amp = []
        self.noise_std = []
        self.det_snr = []
        self.processed = False
        self.pulse_number = pulse_number


# we will use the inj_obj class
class det_stats:
    def __init__(self, **kwargs):
        # this item should contain
        # list: filfiles
        # list: inj_samp
        print("creating class and updating kwargs")
        self.__dict__.update(kwargs)
        # try to access the attribute, throw an exception if not available
        self.filfiles
        self.toas
        self.dms
        self.mask_fn
        self.period
        self.create_burst()

    def create_burst(self):
        temp = []
        for i,(f, m, t, d) in enumerate(zip(self.filfiles, self.mask_fn, self.toas, self.dms)):
            t = det_obj(
                snr=1, toas=t, dm=d, downsamp=self.downsamp, filfile=f, mask=m, pulse_number=i
            )
            temp.append(t)
        self.sorted_pulses = temp

    def calculate_snr(self, multiprocessing=False,manual=True):
        import copy
        plot_folder = "fit_plots"
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)

        if multiprocessing:

            def run_calc(s):
                plot_name = f"{plot_folder}/{s.pulse_number}_{s.filfile.split('/')[-1].split('.')[0]}_{s.toas}"
                s.calculate_fluence_single(period=self.period, manual=manual,plot_name=plot_name)
                return copy.deepcopy(s)

            # for faster debugging
            # self.sorted_pulses = self.sorted_pulses[0:10]
            with ProcessPool(nodes=2) as p:
                self.sorted_pulses = p.map(run_calc, self.sorted_pulses)

        else:
            for i,s in enumerate(self.sorted_pulses):
                plot_name = f"{plot_folder}/{s.pulse_number}_{s.filfile.split('/')[-1].split('.')[0]}_{s.toas}"
                if s.processed:
                    print("already processed, skipping")
                    continue
                print(i,"out of ",len(self.sorted_pulses))
                s.calculate_fluence_single(period = self.period,manual=manual,plot_name=plot_name)
                #dump every 30 pulses
                if i%100 == 0:
                    with open(f"tmp.dill", "wb") as of:
                        dill.dump(inject_stats, of)

    def get_bad_bursts(self,refit="refit"):
        #get all png files in the refit directory
        listdir = os.listdir(refit)
        pngs = [f for f in listdir if f.endswith(".png")]
        pulse_numbers = [f.split("_")[0] for f in pngs]
        for pulse_number in pulse_numbers:
            for s in self.sorted_pulses:
                if s.pulse_number == int(pulse_number):
                    s.processed = False
                    s.det_amp = -1
                    s.noise_std = -1
                    s.det_std = -1
                    s.det_snr = -1
                    s.det_fluence = -1
                    s.fluence_amp = -1

    def calculate_snr_refit(self):
        plot_folder = "fit_plots"
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)

        for i,s in enumerate(self.sorted_pulses):
            plot_name = f"{plot_folder}/{s.pulse_number}_{s.filfile.split('/')[-1].split('.')[0]}_{s.toas}"
            if i%10 == 0:
                with open(f"tmp.dill", "wb") as of:
                    dill.dump(inject_stats, of)
            if s.processed:
                print("already processed, skipping")
                continue
            print(i,"out of ",len(self.sorted_pulses))
            s.calculate_fluence_single(period = self.period,manual=True,plot_name=plot_name)
            #move the refitted png to the original fit_plots folder
            os.system(f"mv refit/{s.pulse_number}_{s.filfile.split('/')[-1].split('.')[0]}_{s.toas}_autofit.png {plot_name}.png")
            #every 20 pulses, save the file

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
    )
    parser.add_argument(
        "-o", default="detection_statistics", help="output file name", required=True
    )
    parser.add_argument("-p", nargs="+", help="TOA files")
    parser.add_argument(
        "-ds", type=int, help="The downsample when getting det_snr", required=True
    )
    parser.add_argument("-period", type=float, help="The period of the pulsar", default=2.0)
    parser.add_argument("-manual", action="store_true", help="Use manual SNR calculation")
    parser.add_argument("-multiprocessing", action="store_true", help="Use multiprocessing")
    parser.add_argument("-checkpoint", default="tmp.dill", help="checkpoint file or file to refit")
    args = parser.parse_args()

    dm = float(args.dm)
    filfiles = []
    maskfiles = []
    # in the cut out the pulse is always at 3s
    positive_fl = args.p
    period = args.period
    downsamp = args.ds
    fil1 = []
    dm1 = []
    toa1 = []
    from read_positive_burst import read_positive_burst

    for p in positive_fl:
        dm_temp, toa_temp, boxcar_det_snr, MJD, fil_temp = read_positive_burst(p)
        # fil_temp,dm_temp,toa_temp = read_positive_file(p)
        if len(fil1) == 0:
            fil1, dm1, toa1 = (fil_temp, dm_temp, toa_temp)
        else:
            fil1, dm1, toa1 = combine_positives(
                fil1, fil_temp, dm1, dm_temp, toa1, toa_temp
            )
        print(len(fil1), len(dm1), len(toa1))
    if dm != 0:
        dm1 = np.array(dm1)
        dm1[:] = dm
    for f in fil1:
        if ".fil" in f:
            filfiles.append(f)
            maskedfn = f.strip(".fil") + "_rfifind.mask"
            maskfiles.append(maskedfn)

    init_obj = {
        "filfiles": fil1,
        "dms": dm1,
        "toas": toa1,
        "mask_fn": maskfiles,
        "downsamp": downsamp,
        "period": period,
    }
    inject_stats = det_stats(**init_obj)
    #check if tmp.dill exists, if so, load it and continue
    if os.path.exists(args.checkpoint):
        with open(args.checkpoint,"rb") as of:
            inject_stats = dill.load(of)
    if args.manual:
        inject_stats.get_bad_bursts()
        inject_stats.calculate_snr_refit()
    else:
        inject_stats.calculate_snr(manual=args.manual,multiprocessing=args.multiprocessing)

    with open(f"{args.o}.dill", "wb") as of:
        dill.dump(inject_stats, of)
    #remove tmp.dill
    #check if tmp.dill exists, if so, load it and continue
    if os.path.exists("tmp.dill"):
        os.remove("tmp.dill")
    #make a refit folder if it doesn't exist
    if not os.path.exists("refit"):
        os.mkdir("refit")
