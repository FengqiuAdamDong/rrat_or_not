#!/usr/bin/env python3
import numpy as np
from sigpyproc import readers as r
try:
    from presto.filterbank import FilterbankFile
    from presto import filterbank as fb
    from presto import rfifind
except:
    print("no presto installed, may fail later")
from matplotlib import pyplot as plt
import sys
from pathos.pools import ProcessPool
import dill
import scipy.optimize as opt
from scipy.optimize import minimize
import argparse
import scipy.optimize as opt
from scipy.optimize import minimize
from gaussian_fitter import log_likelihood
from gaussian_fitter import gaussian
from matplotlib.widgets import Slider, Button, RadioButtons
import copy

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
    print('loading mask')
    rfimask = rfifind.rfifind(maskfn)
    print('getting mask')
    mask = get_mask(rfimask, start_bin, nbinsextra)[::-1]
    print('get mask finished')
    masked_chans = mask.all(axis=1)
    #mask the data but set to the mean of the channel
    mask_vals = np.median(data,axis=1)
    for i in range(len(mask_vals)):
        _ = data[i,:]
        _m = mask[i,:]
        _[_m] = mask_vals[i]
        data[i,:] = _
    return data, masked_chans

def grab_spectra_manual(gf,ts,te,mask_fn,dm,mask=True,downsamp=4,subband=256,manual=False):
    #load the filterbank file
    g = r.FilReader(gf)
    if ts<0:
        ts=0
    print('start and end times',ts,te)
    tsamp = float(g.header.tsamp)
    nsamps = int((te-ts)/tsamp)
    nsamps = nsamps-nsamps%downsamp
    ssamps = int(ts/tsamp)
    #sampels to burst
    nsamps_start_zoom = int(4.1/tsamp)
    nsamps_end_zoom = int(5.9/tsamp)
    spec = g.read_block(ssamps,nsamps)
    #load mask
    if mask:
        print("masking data")
        data, masked_chans = maskfile(mask_fn,spec,ssamps,nsamps)
        print(1024-sum(masked_chans))
    #data.subband(256,subdm=dm,padval='median')
    data = data.dedisperse(dm)
    data = data.downsample(int(downsamp))
    # data.subband(int(subband))
    # data = data.scaled(False)
    ds_data = copy.deepcopy(data)
    # ds_data = ds_data.scaled(False)
    # ds_data.subband(subband)
    ds_data_zoom = ds_data
    ds_data_zoom = ds_data_zoom[:,int(nsamps_start_zoom/downsamp):int(nsamps_end_zoom/downsamp)]
    dat_ts = np.mean(ds_data_zoom[~masked_chans,:],axis=0)
    #make a copy to plot the waterfall
    waterfall_dat = copy.deepcopy(data)
    waterfall_dat = waterfall_dat.downsample(tfactor=1,ffactor=4)
    waterfall_dat = waterfall_dat.normalise()
    waterfall_dat = waterfall_dat[:,int(nsamps_start_zoom/downsamp):int(nsamps_end_zoom/downsamp)]

    if manual:
        SNR,amp,std = fit_SNR_manual(dat_ts,tsamp,2e-2,nsamps=int(0.9/tsamp/downsamp),ds_data=waterfall_dat)
    else:
        SNR,amp,std = fit_SNR(dat_ts,tsamp,2e-2,nsamps=int(0.9/tsamp/downsamp),ds_data=waterfall_dat)

    print("filename:",gf,"downsample:",downsamp,"SNR:",SNR,"amp:",amp,"std",std)
    return SNR,amp,std

def fit_SNR(ts,tsamp,width,nsamps,ds_data):
    #calculates the SNR given a timeseries
    ind_max = nsamps
    w_bin = width/tsamp
    ts_std = np.delete(ts,range(int(ind_max-w_bin),int(ind_max+w_bin)))
    # ts_std = ts
    mean = np.median(ts_std)
    std = np.std(ts_std-mean)
    #subtract the mean
    ts_sub = ts-mean
    #remove rms
    #fit this to a gaussian using ML
    mamplitude = np.max(ts_sub)
    max_ind = nsamps
    #x axis of the fit
    xind = np.array(list(range(len(ts_sub))))

    max_l = minimize(log_likelihood,[mamplitude,max_ind,2,0],args=(xind,ts_sub,std),method='Nelder-Mead')
    fitx = max_l.x
    #there's a negative positive degeneracy
    fitx[0] = abs(fitx[0])
    fitx[1] = abs(fitx[1])
    fitx[2] = abs(fitx[2])
    #residual
    ts_sub = ts_sub - gaussian(xind,fitx[0],fitx[1],fitx[2],fitx[3])
    # plt.figure()
    # plt.plot(ts_sub)
    # plt.show()
    std = np.std(ts_sub)
    Amplitude = fitx[0]
    snr = Amplitude/std
    return snr,Amplitude,std

def fit_SNR_manual(ts,tsamp,width,nsamps,ds_data):
    #calculates the SNR given a timeseries
    ind_max = nsamps
    w_bin = width/tsamp
    ts_std = np.delete(ts,range(int(ind_max-w_bin),int(ind_max+w_bin)))
    # ts_std = ts
    mean = np.median(ts_std)
    std = np.std(ts_std-mean)
    #subtract the mean
    ts_sub = ts-mean
    #remove rms
    #fit this to a gaussian using ML
    mamplitude = np.max(ts_sub)
    max_ind = nsamps
    #x axis of the fit
    xind = np.array(list(range(len(ts_sub))))

    max_l = minimize(log_likelihood,[mamplitude,max_ind,2,0],args=(xind,ts_sub,std),method='Nelder-Mead')
    fitx = max_l.x
    y_fit = gaussian(xind,fitx[0],fitx[1],fitx[2],fitx[3])
    fig=plt.figure(figsize=(50,50))
    # fig=plt.figure(figsize=(5,5))
    ax1 = plt.subplot(1,2,1)
    cmap=plt.get_cmap('magma')
    plt.imshow(ds_data,aspect='auto',cmap=cmap)
    ax = plt.subplot(1,2,2)

    k = plt.plot(xind,ts_sub)
    my_plot, = plt.plot(xind,y_fit,lw=5)
    # ax.margins(x=0)
    axcolor = 'lightgoldenrodyellow'
    pl_ax = plt.axes([0.1, 0.05, 0.78, 0.03], facecolor=axcolor)
    p_ax = plt.axes([0.1, 0.1, 0.78, 0.03], facecolor=axcolor)
    w_ax = plt.axes([0.1, 0.15, 0.78, 0.03], facecolor=axcolor)
    pl = Slider(pl_ax, 'peak loc',0.0 , len(ts), valinit=fitx[1], valstep=1)
    p = Slider(p_ax, 'peak', 0.0, 1, valinit=fitx[0],valstep=1e-5)
    w = Slider(w_ax, 'width', 0.0,100, valinit=fitx[2],valstep=1)
    but_ax = plt.axes([0.1, 0.2, 0.3, 0.03], facecolor=axcolor)
    skipb = Button(but_ax,"Skip")
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    global x_new
    x_new = [-1,-1,-1,-1]
    def update(val):
        peak_loc = pl.val
        peak = p.val
        sigma = w.val
        a = np.mean(ts_sub)
        #refit with new values
        max_l = minimize(log_likelihood,[peak,peak_loc,sigma,a],args=(xind,ts_sub,std),method='Nelder-Mead')
        for i,v in enumerate(max_l.x):
            x_new[i] = v

        print('new fit: ',x_new)
        new_fit = gaussian(xind,x_new[0],x_new[1],x_new[2],x_new[3])
        my_plot.set_ydata(new_fit)
        fig.canvas.draw_idle()

    class skip_class:
        skip=False
        def skip_event(self,event):
            self.skip=True
            plt.close()

    skip = skip_class()
    skipb.on_clicked(skip.skip_event)
    pl.on_changed(update)
    p.on_changed(update)
    w.on_changed(update)
    plt.show()
    if skip.skip:
        print("skipping")
        return -1,-1,-1

    if x_new!=[-1,-1,-1,-1]:
        fitx = x_new
        print("Reassigning fit x becase we've recalculated")
    else:
        print('No refitting done')
    # print("new fit" + str(fitx))
    #there's a negative positive degeneracy
    fitx[0] = abs(fitx[0])
    fitx[1] = abs(fitx[1])
    fitx[2] = abs(fitx[2])
    #residual
    ts_sub = ts_sub - gaussian(xind,fitx[0],fitx[1],fitx[2],fitx[3])
    # plt.figure()
    # plt.plot(ts_sub)
    # plt.show()
    std = np.std(ts_sub)
    Amplitude = fitx[0]
    snr = Amplitude/std
    return snr,Amplitude,std

def logistic(x,k,x0):
    L=1
    return L/(1+np.exp(-k*(x-x0)))

class inject_obj():
    def __init__(self,snr=1,toas=1,dm=1,downsamp=8,filfile="",mask=""):
        self.snr = snr
        self.toas = toas
        self.dm = dm
        self.filfile = filfile
        self.mask = mask
        self.det_snr = []
        self.det_amp = []
        self.det_std = []
        self.downsamp = downsamp

    def repopulate(self, **kwargs):
        self.__dict__.update(kwargs)
        if ~hasattr(self,"detected"):
            self.detected = np.full(len(self.toas),False)


    def calculate_snr_single(self,mask=True):
        ts = self.toas-5
        te = self.toas+5
        snr,amp,std = grab_spectra_manual(gf=self.filfile,ts=ts,te=te,mask_fn=self.mask,dm=self.dm,subband=256,mask=True,downsamp = self.downsamp,manual=True)
        # print(f"Calculated snr:{snr} A:{amp} S:{std} Nominal SNR:{self.snr}")
        self.det_snr = snr
        self.det_amp = amp
        self.det_std = std
        print(snr,amp,std,self.filfile)

    def calculate_snr(self):
        for t,dm in zip(self.toas,self.dm):
            ts = t-5
            te = t+5
            snr,amp,std = grab_spectra_manual(gf=self.filfile,ts=ts,te=te,mask_fn=self.mask,dm=dm,subband=256,mask=True,downsamp = self.downsamp)
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
        inj_data = np.load(self.inj_samp)['grid']
        #first column time stamp, second is snr, third column is dm
        self.downsamp = np.load(self.inj_samp)["downsamp"]
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
            snr = np.round(float(snr_str),2)
            snr_ind = np.round(self.snr_arr,2)==snr
            cur_toa = self.toa_arr[snr_ind]
            cur_dm = self.dm_arr[snr_ind]
            self.sorted_inject.append(inject_obj(snr,cur_toa,cur_dm,self.downsamp,f,m))
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
            det_snr.append(np.median(s.det_snr))
            det_snr_std.append(np.std(s.det_snr))
            inj_snr.append(s.snr)
        plt.errorbar(inj_snr,det_snr,det_snr_std,fmt='.')
        plt.xlabel("Injected SNR")
        plt.ylabel("Detected SNR")
        plt.show()

    def detected_truth(self,si,truth_arr):
        # if we have detected truth array then or the thing, if not then create
        if hasattr(si,"detected"):
            si.detected = (si.detected|truth_arr)
        else:
            si.detected = truth_arr


    def compare(self, fn):
        from read_positive_burst import read_positive_burst_inj
        matched = np.zeros(len(self.dm_arr))
        time_tol = 0.5
        dm_tol = 10
        snr_tol = 1e-2
        for csv in fn:
            dm,burst_time,boxcar_det_snr,inj_snr,MJD = read_positive_burst_inj(csv)
            for t,d,inj_s in zip(burst_time,dm,inj_snr):
                #here we gotta match the values
                t_low = t-time_tol
                t_hi = t+time_tol
                dm_low = d-dm_tol
                dm_hi = d+dm_tol
                # print(t_low,t_hi,dm_low,dm_hi)
                for si in self.sorted_inject:
                    t_snr = (si.snr>(inj_s-snr_tol)) & (si.snr<(inj_s+snr_tol))
                    if t_snr:
                        dm_arr = si.dm
                        t_arr = si.toas
                        truth_dm = (dm_arr<dm_hi)&(dm_arr>dm_low)
                        truth_t = (t_arr<t_hi)&(t_arr>t_low)
                        total_truth = truth_dm & truth_t
                        self.detected_truth(si,total_truth)

        #get lists to plot
        snr = []
        det_frac = []
        for s in self.sorted_inject:
            snr.append(s.snr)
            det_frac.append(sum(s.detected)/len(s.detected))
        #sort snr
        snr = np.array(snr)
        det_frac = np.array(det_frac)
        ind = np.argsort(snr)
        snr = snr[ind]
        det_frac = det_frac[ind]
        self.fit_det(det_frac,snr)
        plt.scatter(snr,det_frac,marker="X")
        plt.show()

    def return_detected(self):
        snr = []
        det = []
        tot = []
        for s in self.sorted_inject:
            snr.append(s.snr)
            det.append(sum(s.detected))
            tot.append(len(s.detected))
        return snr,det,tot


    def fit_det(self,p,snr,plot=True):
        popt,pcov = opt.curve_fit(logistic,snr,p,[9.6,2.07],maxfev=int(1e6))
        self.logisitic_params = popt
        np.save('det_fun_params',popt)
        plt.plot(snr,logistic(snr,popt[0],popt[1]))
        plt.xlabel('SNR')
        plt.ylabel('Detection percentage')

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", action='store_false', default = True, help="Set to do inj_stats analysis")
    parser.add_argument('-l', nargs='+', help='list of filterbank files or positive burst csv files', required=True)
    args = parser.parse_args()
    do_snr_calc = args.d
    if do_snr_calc:
        inj_samples = 'sample_injections.npz'
        filfiles = args.l
        init = {'filfiles':filfiles,'inj_samp':inj_samples}
        inj_stats = inject_stats(**init)
        inj_stats.load_inj_samp()
        inj_stats.match_inj()
        print(len(inj_stats.toa_arr))
        inj_stats.calculate_snr(False)
        with open('inj_stats.dill','wb') as of:
            dill.dump(inj_stats,of)
    else:
        with open('inj_stats.dill','rb') as inf:
            inj_stats = dill.load(inf)
        fns = args.l
        inj_stats = inject_stats(**inj_stats.__dict__)
        inj_stats.repopulate_io()
        inj_stats.calculate_snr_statistics()
        inj_stats.compare(fns)
        with open('inj_stats_fitted.dill','wb') as of:
            dill.dump(inj_stats,of)
