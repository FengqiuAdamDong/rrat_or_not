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
                inst.compare([csv],title=f)
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
        fit_det(det_frac,snrs)
        plt.savefig("overall_selection.png")

def logistic(x,k,x0):
    L=1
    return L/(1+np.exp(-k*(x-x0)))

def fit_det(p,snr,plot=True):
    import scipy.optimize as opt
    popt,pcov = opt.curve_fit(logistic,snr,p,[9.6,2.07],maxfev=int(1e6))
    logisitic_params = popt
    np.save('det_fun_params',popt)
    if plot:
        plt.plot(snr,logistic(snr,popt[0],popt[1]))
        plt.xlabel('SNR')
        plt.ylabel('Detection percentage')

def combine_images():
    import os
    import glob
    from PIL import Image
    image_array = glob.glob("*detection_curve.png")
    images = [Image.open(x) for x in image_array]
    widths, heights = zip(*(i.size for i in images))
    row_len = 10
    total_width = widths[0]*row_len
    max_height = (int(len(heights)/row_len)+1)*heights[0]

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    y_offset = 0
    for i,im in enumerate(images):
        new_im.paste(im, (x_offset,y_offset))
        if (i>0)&((i%row_len)==0):
            y_offset += im.size[1]
            x_offset = 0
        else:
            x_offset += im.size[0]

    new_im.save('detection_curves_all_combined.jpg')

#All inputs are
if __name__=="__main__":
    fil_files = sys.argv[1:]
    inj_collection = inject_stats_collection()
    for i,f in enumerate(fil_files):
        folder_name = f.replace(".fil","")
        try:
            with open(folder_name+'/inj_stats.dill','rb') as inf:
                inj_stats = dill.load(inf)
                inj_stats = inject_stats(**inj_stats.__dict__)
                inj_stats.repopulate_io()

                inj_collection.inj_stats.append(inj_stats)
                inj_collection.folder.append(folder_name)
        except Exception as e:
            #for whatever reason, this failed, lets write it out and just move on
            print(f"failed on {f} with error {e}")
            continue

    inj_collection.calculate_detection_curve()
    combine_images()
