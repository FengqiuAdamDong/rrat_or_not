#!/usr/bin/env python3

###THIS SCRIPT WILL go through every file and combine the inj_stats to give you an overall inj_stats
import numpy as np
import matplotlib.pyplot as plt
import sys
import dill
from inject_stats import inject_stats
class inject_stats_collection:
    def __init__(self):
        self.inj_stats = []
        self.folder = []

    def calculate_detection_curve(self,csvs="1"):
        #build statistics
        snrs = []
        detecteds = []
        totals = []
        for inst,f in zip(self.inj_stats,self.folder):
            if csvs!="all":
                #only compare csv_1
                csv = f"{f}/positive_bursts_1.csv"
                inst.compare([csv])
                snr,det,tot = inst.return_detected()
                for s,d,t in zip(snr,det,tot):
                    if s in snrs:
                        i = np.argwhere(np.array(snrs)==s)[0][0]
                        detecteds[i] = detecteds[i]+d
                        totals[i] = totals[i]+t
                    else:
                        snrs.append(s)
                        detecteds.append(d)
                        totals.append(t)
        detecteds = np.array(detecteds)
        totals = np.array(totals)
        snrs = np.array(snrs)
        det_frac = detecteds/totals
        plt.scatter(snrs,det_frac)
        plt.show()


#All inputs are
if __name__=="__main__":
    fil_files = sys.argv[1:]
    inj_collection = inject_stats_collection()
    for i,f in enumerate(fil_files):
        folder_name = f.replace(".fil","")
        with open(folder_name+'/inj_stats.dill','rb') as inf:
            inj_stats = dill.load(inf)
            inj_stats = inject_stats(**inj_stats.__dict__)
            inj_stats.repopulate_io()

            inj_collection.inj_stats.append(inj_stats)
            inj_collection.folder.append(folder_name)
    inj_collection.calculate_detection_curve()
    import pdb; pdb.set_trace()
