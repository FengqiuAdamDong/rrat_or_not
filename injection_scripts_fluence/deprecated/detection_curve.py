#!/usr/bin/env python3
 #!/usr/bin/env python3
import numpy as np
try:
    from presto.filterbank import FilterbankFile
    from presto import filterbank as fb
    from presto import rfifind
except:
    print("no presto installed, may fail later")
from matplotlib import pyplot as plt
import sys
import scipy.optimize as opt
from scipy.optimize import minimize
import argparse
import scipy.optimize as opt
from scipy.optimize import minimize
from gaussian_fitter import log_likelihood
from gaussian_fitter import gaussian

def logistic(x,k,x0):
    L=1
    return L/(1+np.exp(-k*(x-x0)))

class inject_obj():
    def __init__(self,snr=1,toas=1,dm=1,filfile="",mask=""):
        self.snr = snr
        self.toas = toas
        self.dm = dm
        self.filfile = filfile
        self.mask = mask
        self.det_snr = []
        self.det_amp = []
        self.det_std = []
    def repopulate(self, **kwargs):
        self.__dict__.update(kwargs)

    def return_detected(self):
        #returns the detected snrs
        return self.det_snr[self.detected]
        # pass

class inject_stats():
    def __init__(self, **kwargs):
        #this item should contain
        #list: filfiles
        #list: inj_samp
        print("creating class and updating kwargs")
        self.__dict__.update(kwargs)
        #try to access the attribute, throw an exception if not available
        self.filfiles
        self.inj_samp

    def repopulate_io(self, ):
        if hasattr(self,"sorted_inject"):
            #repopulate sorted inject
            temp = []
            for s in self.sorted_inject:
                t = inject_obj()
                t.repopulate(**s.__dict__)
                temp.append(t)
            self.sorted_inject = np.array(temp)
    def load_inj_samp(self):
        inj_data = np.load(self.inj_samp)
        #first column time stamp, second is snr, third column is dm
        self.toa_arr = inj_data[:,0]
        self.snr_arr = inj_data[:,1]
        self.dm_arr = inj_data[:,2]

    def match_inj(self, ):
        #match snr,toa,dm with certain fil file
        #This function creates the sorted inject objects.
        self.sorted_inject = []
        for snr,dm in zip(self.snr_arr, self.dm_arr):
            self.sorted_inject.append(inject_obj(snr,self.toa_arr,dm))
        self.sorted_inject = np.array(self.sorted_inject)

    def detected_truth(self,si,truth_arr):
        # if we have detected truth array then or the thing, if not then create
        if hasattr(si,"detected"):
            si.detected = (si.detected|truth_arr)
        else:
            si.detected = truth_arr


    def compare(self, fn):
        from automated_period import get_burst_dict
        matched = np.zeros(len(self.dm_arr))
        time_tol = 0.5
        dm_tol = 10
        snr_tol = 1e-2
        for csv in fn:
            mjd,burst_time,subband,dm,snr = get_burst_dict(csv)
            for t,s,d in zip(burst_time,snr,dm):
                #here we gotta match the values
                t_low = t-time_tol
                t_hi = t+time_tol
                dm_low = d-dm_tol
                dm_hi = d+dm_tol
                s_low = s-snr_tol
                s_hi = s+snr_tol
                # print(t_low,t_hi,dm_low,dm_hi)
                for si in self.sorted_inject:
                    dm_arr = si.dm
                    t_arr = si.toas
                    snr_arr = np.zeros_like(t_arr)+si.snr
                    truth_dm = (dm_arr<dm_hi)&(dm_arr>dm_low)
                    truth_t = (t_arr<t_hi)&(t_arr>t_low)
                    s_truth = (snr_arr<s_hi)&(snr_arr>s_low)
                    total_truth = truth_dm&truth_t&s_truth
                    self.detected_truth(si,total_truth)
        #get lists to plot
        #note here det refers to detected by the pipeline
        self.det_snr_arr = []
        self.det_fraction = []
        for si in self.sorted_inject:
            self.det_snr_arr.append(si.snr)
            self.det_fraction.append(sum(si.detected)/len(si.detected))
        #order things
        self.det_snr_arr = np.array(self.det_snr_arr)
        self.det_fraction = np.array(self.det_fraction)
        order_index = np.argsort(self.det_snr_arr)
        self.det_snr_arr = self.det_snr_arr[order_index]
        self.det_fraction = self.det_fraction[order_index]

    def fit_det(self,plot=True):
        #fit the detection curve
        popt,pcov = opt.curve_fit(logistic,self.det_snr_arr,self.det_fraction,[9.6,1.07])
        self.logisitic_params = popt
        np.save('det_fun_params',popt)
        if plot:
            plt.plot(self.det_snr_arr,logistic(self.det_snr_arr,popt[0],popt[1]))
            plt.scatter(self.det_snr_arr,self.det_fraction)
            plt.xlabel('SNR')
            plt.ylabel('Detection fraction')
            plt.show()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', default = [], nargs='+', help='list of positive burst csv files')
    parser.add_argument('-f', default = [], nargs='+', help='list of filterbank files')

    args = parser.parse_args()
    inj_samples = 'sample_injections.npy'
    filfiles = args.f
    init = {'filfiles':filfiles,'inj_samp':inj_samples}
    inj_stats = inject_stats(**init)
    inj_stats.load_inj_samp()
    inj_stats.match_inj()
    fns = args.l

    inj_stats.compare(fns)
    inj_stats.fit_det()
    print(inj_stats.logisitic_params)
    np.save('det_fun_params',inj_stats.logisitic_params)
        # with open('inj_stats_fitted.dill','wb') as of:
            # dill.dump(inj_stats,of)
