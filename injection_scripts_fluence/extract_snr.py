#!/usr/bin/env python3
import argparse
import numpy as np
from inject_stats import inject_obj
from inject_stats import get_mask_fn
from pathos.pools import ProcessPool
import dill
import sys
import csv


class det_obj(inject_obj):
    # det obj is inherited from injection object....
    pass


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
        self.create_burst()

    def create_burst(self):
        temp = []
        for f, m, t, d in zip(self.filfiles, self.mask_fn, self.toas, self.dms):
            t = det_obj(
                snr=1, toas=t, dm=d, downsamp=self.downsamp, filfile=f, mask=m
            )
            temp.append(t)
        self.sorted_pulses = temp

    def calculate_snr(self, multiprocessing=False):
        import copy

        if multiprocessing:

            def run_calc(s):
                s.calculate_fluence_single()
                return copy.deepcopy(s)

            # for faster debugging
            # self.sorted_pulses = self.sorted_pulses[0:10]
            with ProcessPool(nodes=64) as p:
                self.sorted_pulses = p.map(run_calc, self.sorted_pulses)

        else:
            for s in self.sorted_pulses:
                s.calculate_fluence_single()


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
    args = parser.parse_args()

    dm = float(args.dm)
    filfiles = []
    maskfiles = []
    # in the cut out the pulse is always at 3s
    positive_fl = args.p
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
    }
    inject_stats = det_stats(**init_obj)
    inject_stats.calculate_snr()
    with open(f"{args.o}.dill", "wb") as of:
        dill.dump(inject_stats, of)
