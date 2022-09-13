#!/usr/bin/env python3

import numpy as np
from inject_stats import inject_obj
from inject_stats import get_mask_fn
from pathos.pools import ProcessPool
import dill
import sys
import csv
#we will use the inj_obj class
class inject_stats():
    def __init__(self, **kwargs):
        #this item should contain
        #list: filfiles
        #list: inj_samp
        print("creating class and updating kwargs")
        self.__dict__.update(kwargs)
        #try to access the attribute, throw an exception if not available
        self.filfiles
        self.toas
        self.dms
        self.mask_fn
        self.create_burst()

    def create_burst(self):
        temp = []
        for f,m,t,d in zip(self.filfiles,self.mask_fn,self.toas,self.dms):
    # def __init__(self,snr=1,toas=1,dm=1,filfile="",mask=""):
            t = inject_obj(toas=t,dm=d,filfile=f,mask = m)
            temp.append(t)
        self.sorted_pulses = temp
        print(self.sorted_pulses)

    def calculate_snr(self,multiprocessing=False):
        import copy
        if multiprocessing:
            def run_calc(s):
                s.calculate_snr_single(True)
                print(s.det_snr)
                return copy.deepcopy(s)
            #for faster debugging
            # self.sorted_pulses = self.sorted_pulses[0:10]
            with ProcessPool(nodes=64) as p:
                self.sorted_pulses = p.map(run_calc,self.sorted_pulses)
            #plot the snrs
            det_snr = np.array(list(s.det_snr for s in self.sorted_pulses))
            import matplotlib.pyplot as plt
            plt.hist(det_snr,bins=50)
            plt.show()
        else:
            for s in self.sorted_pulses:
                s.calculate_snr_single(True)

def read_positive_file(positive_file):
    filfiles = []
    dms = []
    timestamps = []
    #this function reads the positive bursts files
    with open(positive_file,'r') as csvf:
        p_reader = csv.reader(csvf,delimiter=',')
        for row in p_reader:
            filfiles.append(row[0])
            dms.append(float(row[1]))
            timestamps.append(float(row[2]))
    return filfiles,dms,timestamps

def combine_positives(fil1_,fil2_,dm1_,dm2_,toa1_,toa2_):
    #this function combines two sets (from positive_bursts_1 and positive_bursts_short eg)
    #fil 1 is the one I'm keeping
    fil_add = []
    dm_add = []
    toa_add = []
    no_match = False
    for fil2,dm2,toa2 in zip(fil2_,dm2_,toa2_):
        for fil1,dm1,toa1 in zip(fil1_,dm1_,toa1_):
            no_match = False
            if (fil1==fil2)&(dm1==dm2)&(toa1==toa2):
                break
            no_match = True
        if no_match:
            fil_add.append(fil2)
            dm_add.append(dm2)
            toa_add.append(toa2)
    return np.append(fil1_,fil_add),np.append(dm1_,dm_add),np.append(toa1_,toa_add)

def get_mask_fn_arr(filfiles):
    #get the filenames of all the masks
    mask_fn = [get_mask_fn(f) for f in filfiles]
    print(mask_fn)
    return mask_fn


if __name__=='__main__':
    # fn = 'real_pulses/positive_bursts_edit_snr.csv'
    # fn1 = 'real_pulses/positive_bursts_1_edit_snr.csv'
    # fn2 = 'real_pulses/positive_bursts_short_edit_snr.csv'
    # fn = 'real_pulses/positive_burst_test.csv'
    # fn1 = fn
    # fn2 = fn
    #dedisperse to some nominal DM to make it easier
    dm = 30.9
    import os
    fn = 'real_pulses/positive_bursts_edit_snr.csv'
    fn1 = 'real_pulses/positive_bursts_1_edit_snr.csv'
    fn2 = 'real_pulses/positive_bursts_short_edit_snr.csv'
    fil1,dm1,toa1 = read_positive_file(fn)
    fil2,dm2,toa2 = read_positive_file(fn1)
    fil3,dm3,toa3 = read_positive_file(fn2)
    print(len(fil1),len(dm1),len(toa1))
    fil1,dm1,toa1 = combine_positives(fil1,fil2,dm1,dm2,toa1,toa2)
    print(len(fil1),len(dm1),len(toa1))
    fil1,dm1,toa1 = combine_positives(fil1,fil3,dm1,dm3,toa1,toa3)
    print(len(fil1),len(dm1),len(toa1))
    mask1 = get_mask_fn_arr(fil1)
    print(fil1)
    #just use a nominal dm
    dm1 = np.zeros(len(fil1))+dm
    #in the cut out the pulse is always at 3s



    init_obj = {'filfiles':fil1,'dms':dm1,'toas':toa1,'mask_fn':mask1}
    inject_stats = inject_stats(**init_obj)
    inject_stats.calculate_snr(True)
    with open(f"test.dill",'wb') as of:
        dill.dump(inject_stats,of)
