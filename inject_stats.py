#!/usr/bin/env python3
import numpy as np
#from presto.filterbank import FilterbankFile
#from presto import filterbank as fb
#from presto import rfifind
from matplotlib import pyplot as plt
import sys
from pathos.pools import ProcessPool
import dill
import scipy.optimize as opt

def get_mask_fn(filterbank):
    folder = filterbank.strip('.fil')
    mask = f"{folder}_rfifind.mask"
    return mask

def get_mask(rfimask, startsamp, N):
    """Return an array of boolean values to act as a mask
        for a Spectra object.

        Inputs:
            rfimask: An rfifind.rfifind object
            startsamp: Starting sample
            N: number of samples to read

        Output:
            mask: 2D numpy array of boolean values.
                True represents an element that should be masked.
    """
    sampnums = np.arange(startsamp, startsamp+N)
    blocknums = np.floor(sampnums/rfimask.ptsperint).astype('int')
    mask = np.zeros((N, rfimask.nchan), dtype='bool')
    for blocknum in np.unique(blocknums):
        blockmask = np.zeros_like(mask[blocknums==blocknum])
        chans_to_mask = rfimask.mask_zap_chans_per_int[blocknum]
        if chans_to_mask.any():
            blockmask[:,chans_to_mask] = True
        mask[blocknums==blocknum] = blockmask
    return mask.T

def get_mask_arr(gfb):
    mask_arr = []
    for g in gfb:
        print(g)
        mask_arr.append(get_mask_fn(g))
    return mask_arr

def maskfile(maskfn, data, start_bin, nbinsextra):
    from presto import rfifind
    rfimask = rfifind.rfifind(maskfn)
    mask = get_mask(rfimask, start_bin, nbinsextra)[::-1]
    masked_chans = mask.all(axis=1)
    data = data.masked(mask, maskval='median-mid80')
    return data, masked_chans

def grab_spectra(gf,ts,te,mask_fn,dm):
    #load the filterbank file
    g = FilterbankFile(gf,mode='read')
    tsamp = float(g.header['tsamp'])
    nsamps = int((te-ts)/tsamp)
    ssamps = int(ts/tsamp)
    #sampels to burst
    nsamps_start_zoom = int(2.5/tsamp)
    nsamps_end_zoom = int(3.5/tsamp)
    spec = g.get_spectra(ssamps,nsamps)
    #load mask
    spec.dedisperse(dm, padval='median')
    data, masked_chans = maskfile(mask_fn,spec,ssamps,nsamps)
    #data.subband(256,subdm=dm,padval='median')
    subband = 256
    downsamp = 3
    data.downsample(int(downsamp))
    # data.subband(int(subband))
    # data = data.scaled(False)
    dat_arr = data.data
    dat_arr = dat_arr[~masked_chans,:]
    dat_arr = dat_arr[:,int(nsamps_start_zoom/downsamp):int(nsamps_end_zoom/downsamp)]
    dat_ts = np.mean(dat_arr,axis=0)

    SNR,amp,std = calculate_SNR(dat_ts,tsamp,10e-3,nsamps=int(0.5/tsamp/downsamp))
    return SNR,amp,std
    # plt.plot(ts_sub)
    # plt.title(f"{gf} - SNR:{SNR}")
    # plt.figure()
    # plt.imshow(dat_arr,aspect='auto')
    # plt.show()

def calculate_SNR(ts,tsamp,width,nsamps):
    #calculates the SNR given a timeseries

    ind_max = nsamps
    w_bin = width/tsamp
    ts_std = np.delete(ts,range(int(ind_max-w_bin),int(ind_max+w_bin)))
    # ts_std = ts
    mean = np.median(ts_std)
    std = np.std(ts_std)
    #subtract the mean
    ts_sub = ts-mean
    #remove rms
    Amplitude = ts_sub[nsamps]
    snr = Amplitude/std
    # print(np.mean(ts_sub))
    # plt.plot(ts_std)
    # plt.show()
    #area of pulse, the noise, if gaussian should sum, to 0
    return snr,Amplitude,std

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

    def calculate_snr_single(self):
            ts = self.toas-3
            te = self.toast+3
            snr,amp,std = grab_spectra(self.filfile,ts,te,self.mask,self.dm)
            # print(f"Calculated snr:{snr} A:{amp} S:{std} Nominal SNR:{self.snr}")
            self.det_snr = snr
            self.det_amp = amp
            self.det_std = std

    def calculate_snr(self):
        for t,dm in zip(self.toas,self.dm):
            ts = t-3
            te = t+3
            snr,amp,std = grab_spectra(self.filfile,ts,te,self.mask,dm)
            # print(f"Calculated snr:{snr} A:{amp} S:{std} Nominal SNR:{self.snr}")
            self.det_snr.append(snr)
            self.det_amp.append(amp)
            self.det_std.append(std)
        self.det_snr = np.array(self.det_snr)
        self.det_amp = np.array(self.det_amp)
        self.det_std = np.array(self.det_std)

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
        if not hasattr(self,'mask_fn'):
            self.get_mask_fn()
    def repopulate_io(self, ):
        if hasattr(self,"sorted_inject"):
            #repopulate sorted inject
            temp = []
            for s in self.sorted_inject:
                t = inject_obj()
                t.repopulate(**s.__dict__)
                temp.append(t)
            self.sorted_inject = np.array(temp)

    def get_mask_fn(self):
        #get the filenames of all the masks
        self.mask_fn = [get_mask_fn(f) for f in self.filfiles]

    def load_inj_samp(self):
        inj_data = np.load(self.inj_samp)
        #first column time stamp, second is snr, third column is dm
        self.toa_arr = inj_data[:,0]
        self.snr_arr = inj_data[:,1]
        self.dm_arr = inj_data[:,2]

    def match_inj(self, ):
        #match snr,toa,dm with certain fil file
        self.sorted_inject = []
        for f,m in zip(self.filfiles, self.mask_fn):
            sp_ = f.split('_')
            snr_str = sp_[-1]
            snr_str = snr_str.strip('.fil').strip('snr')
            snr = np.round(float(snr_str),3)
            snr_ind = np.round(self.snr_arr,3)==snr
            cur_toa = self.toa_arr[snr_ind]
            cur_dm = self.dm_arr[snr_ind]
            self.sorted_inject.append(inject_obj(snr,cur_toa,cur_dm,f,m))
        self.sorted_inject = np.array(self.sorted_inject)

    def calculate_snr(self,multiprocessing=False):
        import copy
        if multiprocessing:
            def run_calc(s):
                s.calculate_snr()
                print(s.det_snr)
                return copy.deepcopy(s)
            #for faster debugging
            # self.sorted_inject = self.sorted_inject[0:10]
            with ProcessPool(nodes=64) as p:
                self.sorted_inject = p.map(run_calc,self.sorted_inject)

        else:
            for s in self.sorted_inject:
                s.calculate_snr()

    def calculate_snr_statistics(self):
        det_snr = []
        det_snr_std = []
        inj_snr = []
        for s in self.sorted_inject:
            det_snr.append(np.mean(s.det_snr))
            det_snr_std.append(np.std(s.det_snr))
            inj_snr.append(s.snr)
        plt.errorbar(inj_snr,det_snr,det_snr_std,fmt='.')
        plt.show()

    def detected_truth(self,si,truth_arr):
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
        det_snr_arr = []
        total_snr_arr = []
        for s in self.sorted_inject:
            det_snr_arr.append(s.return_detected())
            total_snr_arr.append(s.det_snr)
        det_snr_arr = np.array(list([item for sublist in det_snr_arr for item in sublist]))
        total_snr_arr = np.array(list([item for sublist in total_snr_arr for item in sublist]))
        print(len(det_snr_arr))
        print(len(total_snr_arr))
        bin_widths = np.linspace(-0,5,25)
        hist_det,bin_edges_det = np.histogram(det_snr_arr,bin_widths)
        hist_t, bin_edges_t = np.histogram(total_snr_arr,bin_widths)
        p = hist_det/hist_t
        bin_centres = bin_edges_t+np.diff(bin_edges_t)[0]
        bin_centres = bin_centres[:-1]
        self.fit_det(p,bin_centres)
        plt.plot(bin_centres,p)
        plt.show()
        #now get the bins
        # np.savez('det_curve',snr=u_snr,p=det_frac)
        # plt.plot(u_snr+1,det_frac)
        # plt.xlabel('snr')
        # plt.ylabel('det fraction')
        # plt.show()


    def fit_det(self,p,snr,plot=True):
        popt,pcov = opt.curve_fit(logistic,snr,p,[9.6,1.07])
        print(popt)
        plt.plot(snr,logistic(snr,popt[0],popt[1]))
        plt.xlabel('SNR')
        plt.ylabel('Detection percentage')

if __name__=="__main__":
    do_snr_calc = False
    if do_snr_calc:
        inj_samples = 'sample_injections.npy'
        filfiles = sys.argv[1:]
        init = {'filfiles':filfiles,'inj_samp':inj_samples}
        inj_stats = inject_stats(**init)
        inj_stats.load_inj_samp()
        inj_stats.match_inj()
        print(len(inj_stats.toa_arr))
        inj_stats.calculate_snr(True)
        with open('inj_stats.dill','wb') as of:
            dill.dump(inj_stats,of)
    else:
        with open('inj_stats.dill','rb') as inf:
            inj_stats = dill.load(inf)
        fns = sys.argv[1:]
        inj_stats = inject_stats(**inj_stats.__dict__)
        inj_stats.repopulate_io()
        inj_stats.calculate_snr_statistics()
        inj_stats.compare(fns)
