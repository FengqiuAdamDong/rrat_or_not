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
def create_matrix(x, y, z, norm=1):
    #create the detection matrix
    unique_widths = np.unique(y)
    unique_snrs = np.unique(x)
    det_matrix = np.zeros((len(unique_snrs), len(unique_widths)))
    for i, s in enumerate(unique_snrs):
        for j, w in enumerate(unique_widths):
            ind = (x == s) & (y == w)
            if sum(ind)==0:
                det_matrix[i, j] = np.nan
                continue
            if sum(ind)>1:
                import pdb; pdb.set_trace()
            if norm==1:
                det_matrix[i, j] = z[ind]/w
            elif norm==0:
                det_matrix[i, j] = z[ind]/s
            else:
                det_matrix[i, j] = z[ind]
    return unique_snrs, unique_widths, det_matrix

def get_mask_fn(filterbank):
    folder = filterbank.strip(".fil")
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
    sampnums = np.arange(startsamp, startsamp + N)
    blocknums = np.floor(sampnums / rfimask.ptsperint).astype("int")
    mask = np.zeros((N, rfimask.nchan), dtype="bool")
    for blocknum in np.unique(blocknums):
        blockmask = np.zeros_like(mask[blocknums == blocknum])
        chans_to_mask = rfimask.mask_zap_chans_per_int[blocknum]
        if chans_to_mask.any():
            blockmask[:, chans_to_mask] = True
        mask[blocknums == blocknum] = blockmask
    return mask.T


def get_mask_arr(gfb):
    mask_arr = []
    for g in gfb:
        print(g)
        mask_arr.append(get_mask_fn(g))
    return mask_arr


def maskfile(maskfn, data, start_bin, nbinsextra):
    print("loading mask")
    rfimask = rfifind.rfifind(maskfn)
    print("getting mask")
    mask = get_mask(rfimask, start_bin, nbinsextra)[::-1]
    print("get mask finished")
    masked_chans = mask.all(axis=1)
    # masked_chans = np.load("data.npz")['masked_chans']

    # mask the data but set to the mean of the channel
    # mask_vals = np.median(data, axis=1)
    # for i in range(len(mask_vals)):
    #     _ = data[i, :]
    #     _m = mask[i, :]
    #     _[_m] = mask_vals[i]
    #     data[i, :] = _
    print(f"Masked channels {len(masked_chans)}: {sum(masked_chans)}")

    return data, masked_chans


def extract_plot_data(data,masked_chans,dm,downsamp,nsamps_start_zoom,nsamps_end_zoom):
    # make a copy to plot the waterfall
    waterfall_dat = copy.deepcopy(data)
    #make sure the length of waterfall_dat is a multiple of downsamp
    nsamps = waterfall_dat.shape[1]
    nsamps = nsamps - nsamps % downsamp
    waterfall_dat = waterfall_dat[:,0:nsamps]
    waterfall_dat = waterfall_dat.dedisperse(dm)
    waterfall_dat = waterfall_dat.downsample(tfactor=downsamp)
    dat_ts = copy.deepcopy(waterfall_dat)[~masked_chans, :]
    dat_ts = np.mean(dat_ts, axis=0)
    dat_ts = dat_ts[int(nsamps_start_zoom / downsamp) : int(nsamps_end_zoom / downsamp)]
    #make sure that waterfall_dat is a multiple of 4
    nsamps = waterfall_dat.shape[1]
    nsamps = nsamps - nsamps % 4
    waterfall_dat = waterfall_dat[:,0:nsamps]
    waterfall_dat = waterfall_dat.downsample(ffactor=16, tfactor=4)
    waterfall_dat = waterfall_dat.normalise()
    waterfall_dat = waterfall_dat[
        :, int(nsamps_start_zoom / (downsamp*4)) : int(nsamps_end_zoom / (downsamp*4))
    ]
    return waterfall_dat,dat_ts


def grab_spectra_manual(
        gf, ts, te, mask_fn, dm, mask=True, downsamp=4, subband=256, manual=False, t_start = 4.1, t_dur = 1.8, fit_del = 100e-3,plot_name = "", guess_width = 0.01,
):
    # load the filterbank file
    g = r.FilReader(gf)
    if ts < 0:
        ts = 0
    tsamp = float(g.header.tsamp)
    total_time = float(g.header.nsamples) * tsamp
    if te>total_time:
        te = total_time
        #change t_dur to be the time from 4.1s to end
        t_right = te - ts - t_start
        if t_right<t_dur:
            t_dur = t_right
    if ts<0:
        #shift t_start back by however much time you need
        t_start = t_start + ts*tsamp
        ts = 0


    print("start and end times", ts, te)
    nsamps = int((te - ts) / tsamp)
    nsamps = nsamps - nsamps % downsamp
    ssamps = int(ts / tsamp)
    # sampels to burst
    nsamps_start_zoom = int(t_start / tsamp)
    nsamps_end_zoom = int((t_dur+t_start) / tsamp)
    try:
        spec = g.read_block(ssamps, nsamps)
    except Exception as e:
        print(e)
        import pdb; pdb.set_trace()

    # load mask
    if mask:
        print("masking data")
        data, masked_chans = maskfile(mask_fn, spec, ssamps, nsamps)
    waterfall_dat, dat_ts = extract_plot_data(data,masked_chans,dm,downsamp,nsamps_start_zoom,nsamps_end_zoom)

    if manual:
        # this gives you the location of the peak
        amp, std, loc, sigma_width = fit_SNR_manual(
            dat_ts,
            tsamp * downsamp,
            fit_del,
            nsamps=int(t_dur/2 / tsamp / downsamp),
            ds_data=waterfall_dat,
            downsamp=downsamp,
        )

        if (amp!=-1)&((loc<(0.49*t_dur))|(loc>(t_dur*0.51))|(sigma_width>2e-2)):
            #repeat if initial loc guess is wrong
            amp, std, loc, sigma_width, FLUENCE = fit_SNR_manual(
                dat_ts,
                tsamp * downsamp,
                fit_del,
                nsamps=int(loc / tsamp / downsamp),
                ds_data=waterfall_dat,
                downsamp=downsamp,
            )
        #scale the noise by the number of channels that are not masked
        std = std * np.sqrt(sum(~masked_chans)/len(masked_chans))
        SNR = amp / std
        if amp!=-1:
            loc = loc+t_start
            ts_no_ds_zoom_start = int(loc/tsamp - 0.9/tsamp)
            ts_no_ds_zoom_end = int(loc/tsamp + 0.9/tsamp)
            ts_no_ds = data[:, ts_no_ds_zoom_start : ts_no_ds_zoom_end]
            ts_no_ds = np.mean(ts_no_ds[~masked_chans, :], axis=0)
        else:
            FLUENCE = -1
    else:
        # fit using downsampled values
        # this is mostly used for the injections
        try:
            amp, std, loc, sigma_width, FLUENCE = autofit_pulse(
                dat_ts,
                tsamp * downsamp,
                fit_del,
                nsamps=int(t_dur / 2 / tsamp / downsamp),
                ds_data=waterfall_dat,
                downsamp=downsamp,
                plot=False,
                plot_name=plot_name,
                fit_width_guess = guess_width
            )
            #refit with new initial params
            amp, std, loc, sigma_width, FLUENCE = autofit_pulse(
                dat_ts,
                tsamp * downsamp,
                fit_del,
                nsamps=int(loc / tsamp / downsamp),
                ds_data=waterfall_dat,
                downsamp=downsamp,
                plot=True,
                plot_name=plot_name,
                fit_width_guess = sigma_width
            )
        except Exception as e:
            #print the full traceback
            import traceback
            traceback.print_exc()
            amp, std, loc, sigma_width, FLUENCE = -1, -1, -1, -1, -1
        #scale std by the sqrt of non masked chans
        std = std * np.sqrt(sum(~masked_chans)/len(masked_chans))
        FLUENCE = FLUENCE/std
        SNR = amp / std
        # because loc is predetermined set start and end a predifined spot
        ts_no_ds_zoom_start = int(4.1/tsamp)
        ts_no_ds_zoom_end = int(5.9/tsamp)
        ts_no_ds = data[:, ts_no_ds_zoom_start : ts_no_ds_zoom_end]
        ts_no_ds = np.mean(ts_no_ds[~masked_chans, :], axis=0)
    # FLUENCE = -1
    # recalculate the amplitude given a gaussian pulse shape
    gaussian_amp = FLUENCE / sigma_width / np.sqrt(2 * np.pi)
    print("filename:", gf, "downsample:", downsamp, "FLUENCE:", FLUENCE)
    approximate_toa = g.header.tstart + ((te+ts)/2)/86400
    return FLUENCE, std, amp, gaussian_amp, sigma_width, SNR, approximate_toa

def find_polynomial_fit(x_std, ts_std, order = None):
    rchi2 = 100
    i = 1
    rchi2_arr = []
    poly_arr = []
    coeffs_arr = []
    std_arr = []
    if order is None:
        for i in range(1,10):

            coeffs = np.polyfit(x_std, ts_std, i)
            poly = np.poly1d(coeffs)
            # Calculate the reduced chi2 of the fit
            ts_diff = ts_std - poly(x_std)
            std_arr.append(np.std(ts_diff))
            rchi2 = np.sum((ts_std - poly(x_std)) ** 2 / (np.std(ts_std[1:500]) ** 2)) / (len(ts_std) - i)
            # print("rchi2", rchi2, "i", i)
            rchi2_arr.append(rchi2)
            poly_arr.append(poly)
            coeffs_arr.append(coeffs)
        # find the minimum rchi2
        rchi2_arr = (np.array(rchi2_arr)-1)**2
        ind = np.argmin(rchi2_arr)
        # print(std_arr)
        std_diff = np.abs(np.diff(std_arr))
        # print(std_diff)
        #find where std_diff first is smaller than 0.0003
        try:
            ind_std = np.where(std_diff < 0.0002)[0][0]
            # print(ind_std)
            poly = poly_arr[ind_std+1]
            coeffs = coeffs_arr[ind_std+1]
        except:
            #if it fails just take the middle value
            poly = poly_arr[5]
            coeffs = coeffs_arr[5]
    else:
        coeffs = np.polyfit(x_std, ts_std, order)
        poly = np.poly1d(coeffs)

    # coeffs = np.polyfit(x_std, ts_std, 10)
    # poly = np.poly1d(coeffs)

    return poly, coeffs

def autofit_pulse(ts, tsamp, width, nsamps, ds_data, downsamp, plot=True, plot_name="", fit_width_guess = 0.01):
    # calculates the SNR given a timeseries
    ind_max = nsamps
    w_bin = width / tsamp
    ts_std = np.delete(ts, range(int(ind_max - w_bin), int(ind_max + w_bin)))
    x = np.linspace(0, tsamp * len(ts), len(ts))
    x_std = np.delete(x, range(int(ind_max - w_bin), int(ind_max + w_bin)))
    # ts_std = ts
    poly, coeffs = find_polynomial_fit(x_std, ts_std)
    # subtract the mean
    ts_sub = ts - poly(x)
    ts_std_sub = ts_std - poly(x_std)
    std = np.std(ts_std_sub)
    # remove rms
    # fit this to a gaussian using ML
    mamplitude = np.max(ts_sub)
    max_time = nsamps * tsamp
    # x axis of the fit
    xind = x
    print(f"initial width guess {fit_width_guess}, amp {mamplitude}, max_time {max_time}")
    max_l = minimize(
        log_likelihood,
        [mamplitude, max_time, fit_width_guess, 0],
        args=(xind, ts_sub, std),
        method="Nelder-Mead",
    )
    fitx = max_l.x
    fitx[0] = abs(fitx[0])
    fitx[1] = abs(fitx[1])
    fitx[2] = abs(fitx[2])
    # residual
    Amplitude = fitx[0]
    loc = fitx[1]
    sigma_width = fitx[2]

    SNR = Amplitude / std
    # once we have calculated the location
    if plot:
        print(f"Making plot {plot_name}_autofit.png")
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].plot(x_std, ts_std)
        axs[0, 0].plot(x, poly(x))
        axs[0, 0].set_title("burst_removed")
        axs[0, 1].plot(x_std, ts_std_sub)
        axs[0, 1].set_title("baseline_removed")
        axs[1, 0].plot(x, ts_sub)
        axs[1, 0].set_title("baseline subtracted")
        axs[1, 0].plot(x, gaussian(xind, fitx[0], fitx[1], fitx[2], fitx[3]))
        axs[1, 0].set_title("baseline subtracted")
        axs[1, 1].plot(x, ts)
        axs[1, 1].set_title("OG time series")
        plt.savefig(f"{plot_name}_autofit.png")
        plt.close()
    #after the mean is subtracted calculate fluence ***this fluence is not normalised, ie. we haven't divided by the std
    fluence = np.trapz(ts_sub, x)
    print(f"Fitted loc:{loc} amp:{Amplitude} std:{std} width sigma:{sigma_width} fluence:{fluence}")
    return Amplitude, std, loc, sigma_width, fluence

def fit_SNR_manual(ts, tsamp, width, nsamps, ds_data, downsamp):
    # calculates the SNR given a timeseries
    ind_max = nsamps
    w_bin = width / tsamp
    ts_std = np.delete(ts, range(int(ind_max - w_bin), int(ind_max + w_bin)))
    x = np.linspace(0, tsamp * len(ts), len(ts))
    x_std = np.delete(x, range(int(ind_max - w_bin), int(ind_max + w_bin)))
    # ts_std = ts
    poly, coeffs = find_polynomial_fit(x_std, ts_std)
    # subtract the mean
    ts_sub = ts - np.interp(x,x_std,poly(x_std))
    ts_std_sub = ts_std - poly(x_std)
    std = [np.std(ts_std_sub)]
    # remove rms
    # fit this to a gaussian using ML
    mamplitude = np.max(ts_sub)
    max_time = nsamps * tsamp
    # x axis of the fit

    xind = x
    # print("init values",[mamplitude, max_time, width * 0.1, np.mean(ts_sub)])
    max_l = minimize(
        log_likelihood,
        [mamplitude, max_time, 1e-2, np.mean(ts_sub)],
        args=(xind, ts_sub, std[0]),
        bounds=((1e-4, None), (0, max(x)), (1e-6, 5e-2), (-2, 2)),
        method="Nelder-Mead",
    )
    fitx = max_l.x
    print(fitx)
    # double the resolution
    xind_fit = np.linspace(min(xind), max(xind), len(xind) * 2)
    y_fit = gaussian(xind_fit, fitx[0], fitx[1], fitx[2], fitx[3])
    fig,axes = plt.subplots(1,3,figsize=(10,10))
    cmap = plt.get_cmap("YlGnBu_r")
    axes[0].imshow(ds_data, aspect="auto",extent=[0,max(x),0,1], cmap=cmap)
    #plot the polyfit
    axes[2].plot(x, np.interp(x,x_std,poly(x_std)), lw=5, alpha=0.7)
    axes[2].scatter(x_std,ts_std,alpha=0.5,s=4)
    axes[2].scatter(x,ts,alpha=0.5)

    (k, )  = axes[1].plot(xind, ts_sub)
    (my_plot,) = axes[1].plot(xind_fit, y_fit,'r', lw=3,alpha=0.7)

    #make a copy of ts_sub
    ts_sub_copy = copy.deepcopy(ts_sub)
    xind_copy = copy.deepcopy(xind)
    #average every 4th sample of ts_sub_copy
    #make sure ts_sub_copy has length multiple of 4
    ts_sub_copy = ts_sub_copy[:int(len(ts_sub_copy)/4)*4]
    xind_copy = xind_copy[:int(len(xind_copy)/4)*4]
    ts_sub_copy = np.mean(ts_sub_copy.reshape(-1, 4), axis=1)
    xind_copy = np.mean(xind_copy.reshape(-1, 4), axis=1)
    #plot the downsampled versio
    my_plot2 = axes[1].plot(xind_copy, ts_sub_copy, lw=2,alpha=0.7)

    # ax.margins(x=0)
    axcolor = "lightgoldenrodyellow"
    pl_ax = plt.axes([0.1, 0.05, 0.78, 0.03], facecolor=axcolor)
    p_ax = plt.axes([0.1, 0.1, 0.78, 0.03], facecolor=axcolor)
    w_ax = plt.axes([0.1, 0.15, 0.78, 0.03], facecolor=axcolor)
    fit_order_ax = plt.axes([0.1, 0.2, 0.78, 0.03], facecolor=axcolor)

    pl = Slider(pl_ax, "peak loc", 0.0, np.max(x), valinit=fitx[1], valstep=1e-3)
    p = Slider(p_ax, "peak", 0.0, 1, valinit=fitx[0], valstep=1e-5)
    w = Slider(w_ax, "width", 0.0, 0.05, valinit=fitx[2], valstep=1e-3)
    w = Slider(w_ax, "width", 0.0, 0.05, valinit=fitx[2], valstep=1e-3)
    fit_order = Slider(fit_order_ax, "fit order", 0, 10, valinit=len(coeffs)-1, valstep=1)
    but_ax = plt.axes([0.1, 0.02, 0.3, 0.03], facecolor=axcolor)
    but_save = plt.axes([0.5, 0.02, 0.3, 0.03], facecolor=axcolor)

    skipb = Button(but_ax, "Skip")
    saveb = Button(but_save, "Save")

    # plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    global x_new
    x_new = [-1, -1, -1, -1]

    def update(val):
        poly, coeffs = find_polynomial_fit(x_std, ts_std,fit_order.val)
        # subtract the mean
        ts_sub = ts - np.interp(x,x_std,poly(x_std))
        ts_std_sub = ts_std - poly(x_std)
        std[0] = np.std(ts_std_sub)
        print(f"new std {std}")
        peak_loc = pl.val
        peak = p.val
        sigma = w.val
        a = np.mean(ts_sub)

        # refit with new values
        max_l = minimize(
            log_likelihood,
            [peak, peak_loc, sigma, a],
            bounds=((1e-4, None), (0, max(x)), (1e-6, 5e-2), (-2, 2)),
            args=(xind, ts_sub, std[0]),
            method="Nelder-Mead",
        )
        for i, v in enumerate(max_l.x):
            x_new[i] = v

        # print("new fit: ", x_new)
        new_fit = gaussian(xind_fit, x_new[0], x_new[1], x_new[2], x_new[3])
        my_plot.set_ydata(new_fit)
        k.set_ydata(ts_sub)
        fig.canvas.draw_idle()

    class skip_class:
        skip = False
        def skip_event(self, event):
            self.skip = True
            plt.close()

    def save(val):
        plt.close()

    skip = skip_class()
    skipb.on_clicked(skip.skip_event)
    saveb.on_clicked(save)

    pl.on_changed(update)
    p.on_changed(update)
    w.on_changed(update)
    fit_order.on_changed(update)
    plt.show()
    if skip.skip:
        print("skipping")
        return -1, -1, -1, -1

    if x_new != [-1, -1, -1, -1]:
        fitx = x_new
        print("Reassigning fit x becase we've recalculated")
    else:
        print("No refitting done")
    # print("new fit" + str(fitx))
    # there's a negative positive degeneracy
    fitx[0] = abs(fitx[0])
    fitx[1] = abs(fitx[1])
    fitx[2] = abs(fitx[2])
    Amplitude = fitx[0]
    loc = fitx[1]
    sigma_width = fitx[2]

    SNR = Amplitude / std[0]
    fluence = np.trapz(ts_sub, dx=tsamp )
    # once we have calculated the location
    print(f"Fitted loc:{loc} amp:{Amplitude} std:{std[0]} width sigma:{sigma_width}")
    return Amplitude, std[0], loc, sigma_width, fluence


def logistic(x, k, x0):
    L = 1
    return L / (1 + np.exp(-k * (x - x0)))


def gen_log(x,A,B,C,M,K,v):
    return A+((K-A)/((C+np.exp(-B*(x-M)))**(1/v)))


class inject_obj:
    def __init__(
        self, snr=1, width=1, toas=1, dm=1, downsamp=8, filfile="", mask=""
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
        self.approximate_toa = []
        self.processed = False


    def repopulate(self, **kwargs):
        self.__dict__.update(kwargs)
        #find where nans are in fluence array
        if ~hasattr(self, "detected"):
            self.detected = np.full(len(self.toas), False)

        ind = np.isnan(self.det_fluence)
        self.det_amp[ind] = -10
        self.det_std[ind] = -10
        self.det_fluence[ind] = -10
        self.fluence_amp[ind] = -10


    def calculate_fluence_single(self, mask=True, period = 2,manual=True,plot_name=""):
        ts = self.toas - 5
        te = self.toas + 5
        if period > 1.9:
            t_dur = 1.8
            t_start = 4.1
            fit_del = 15e-2
        else:
            t_dur = (period-0.1)*2
            t_start = 5-(t_dur/2)
            fit_del = t_dur*0.055

        fluence, std, amp, gaussian_amp, sigma_width, det_snr, approximate_toa = grab_spectra_manual(
            gf=self.filfile,
            ts=ts,
            te=te,
            mask_fn=self.mask,
            dm=self.dm,
            subband=256,
            mask=True,
            downsamp=self.downsamp,
            manual=manual,
            t_start = t_start,
            t_dur = t_dur,
            fit_del = fit_del,
            plot_name = plot_name
        )
        # print(f"Calculated fluence:{fluence} A:{amp} S:{std} Nominal FLUENCE:{self.fluence}")
        self.approximate_toa = approximate_toa
        self.det_snr = det_snr
        self.det_fluence = fluence
        self.det_amp = amp
        self.fluence_amp = gaussian_amp
        self.det_std = sigma_width
        self.noise_std = std
        #if we get a negative det_amp then set the processed status to false
        if self.det_amp==-1:
            print("setting process to False")
            self.processed = False
        else:
            print("set processed to true")
            self.processed = True
        print(
            f"fitted_snr {det_snr} std {std} amp {amp}"
        )
        # print(fluence,amp,std,self.filfile)

    def calculate_fluence(self):
        for t, dm, snr, width in zip(self.toas, self.dm, self.snr, self.width):
            ts = t - 5
            te = t + 5
            fluence, std, amp, gaussian_amp, sigma_width, det_snr, approximate_toa = grab_spectra_manual(
                gf=self.filfile,
                ts=ts,
                te=te,
                mask_fn=self.mask,
                dm=dm,
                subband=256,
                mask=True,
                downsamp=self.downsamp,
                guess_width=width,
            )
            print(
                f"inj_snr {snr} fitted_snr {det_snr} std {std} amp {amp}"
            )
            # print(f"Calculated fluence:{fluence} A:{amp} S:{std} Nominal FLUENCE:{self.fluence}")

            self.det_fluence.append(fluence)
            self.det_amp.append(amp)
            self.det_std.append(sigma_width)
            self.fluence_amp.append(gaussian_amp)
            self.noise_std.append(std)
            self.det_snr.append(det_snr)
            self.approximate_toa.append(approximate_toa)
        self.det_fluence = np.array(self.det_fluence)
        self.det_amp = np.array(self.det_amp)
        self.det_std = np.array(self.det_std)
        self.fluence_amp = np.array(self.fluence_amp)
        self.noise_std = np.array(self.noise_std)
        self.det_snr = np.array(self.det_snr)
        self.approximate_toa = np.array(self.approximate_toa)



class inject_stats:
    def __init__(self, **kwargs):
        # this item should contain
        # list: filfiles
        # list: inj_samp
        print("creating class and updating kwargs")
        self.downsamp = 1
        self.filfiles = []
        self.__dict__.update(kwargs)
        # try to access the attribute, throw an exception if not available
        self.filfiles
        self.inj_samp
        if not hasattr(self, "mask_fn"):
            self.get_mask_fn()

    def repopulate_io(
        self,
    ):
        if hasattr(self, "sorted_inject"):
            # repopulate sorted inject
            temp = []
            for s in self.sorted_inject:
                t = inject_obj()
                t.repopulate(**s.__dict__)
                temp.append(t)
            self.sorted_inject = np.array(temp)

    def get_base_fn(self):
        #gets the base filename from the filename structure of filfiles
        splits = self.filfiles[0].split("_")
        #remove last index
        splits = splits[:-1]
        # self.base_fn = "_".join(splits)
    def get_mask_fn(self):
        # get the filenames of all the masks
        self.mask_fn = [get_mask_fn(f) for f in self.filfiles]

    def load_inj_samp(self):
        inj_data = np.load(self.inj_samp)["grid"]
        # first column time stamp, second is snr, third column is dm last column is injected width
        self.toa_arr = inj_data[:, 0]
        self.snr_arr = inj_data[:, 1]
        self.dm_arr = inj_data[:, 2]
        self.width_arr = inj_data[:, 3]

    def match_inj(
        self,
    ):
        # match snr,toa,dm with certain fil file
        self.sorted_inject = []
        for f, m in zip(self.filfiles, self.mask_fn):
            sp_ = f.split("_")
            #find the snr str and width str
            for s in sp_:
                if "SNR" in s:
                    snr_str = s
                if "width" in s:
                    width_str = s
            snr_str = snr_str.strip(".fil").strip("SNR")
            width_str = width_str.strip(".fil").strip("width")
            snr = np.round(float(snr_str), 4)
            width = np.round(float(width_str), 4)
            snr_ind = np.round(self.snr_arr, 4) == snr
            width_ind = np.round(self.width_arr, 4) == width

            cur_ind = snr_ind & width_ind
            cur_snr = self.snr_arr[cur_ind]
            cur_toa = self.toa_arr[cur_ind]
            cur_dm = self.dm_arr[cur_ind]
            cur_width = self.width_arr[cur_ind]
            if len(cur_snr)==0:
                print(f"WARNING NO MATCHING SNR, THIS SHOULD NOT HAPPEN NORMALLY. IF YOU ARE NOT EXPECTING THIS WARNING THEN CHECK THE FILES CREATED {f}")
                continue

            self.sorted_inject.append(
                inject_obj(
                    snr=cur_snr,
                    width=cur_width,
                    toas=cur_toa,
                    dm=cur_dm,
                    downsamp=self.downsamp,
                    filfile=f,
                    mask=m,
                )
            )
        self.sorted_inject = np.array(self.sorted_inject)

    def calculate_snr(self, multiprocessing=False):
        import copy

        if multiprocessing:

            def run_calc(s):
                s.calculate_fluence()
                return copy.deepcopy(s)

            # for faster debugging
            # self.sorted_inject = self.sorted_inject[0:10]
            with ProcessPool(nodes=64) as p:
                self.sorted_inject = p.map(run_calc, self.sorted_inject)

        else:
            for i,s in enumerate(self.sorted_inject):
                print(i,"out of",len(self.sorted_inject))
                s.calculate_fluence()

    def amplitude_statistics(self,title="f"):
        det_snr = []
        inj_snr = []
        det_width = []
        inj_width = []
        det_fluence = []
        inj_fluence =[]
        det_snr_std = []
        det_width_std = []
        det_fluence_std = []
        noise_std = []
        det_amp = []
        det_amp_std = []

        for s in self.sorted_inject:
            det_snr.append(np.mean(s.det_snr))
            det_snr_std.append(np.std(s.det_snr))

            det_width.append(np.mean(s.det_std))
            det_width_std.append(np.std(s.det_std))

            det_fluence.append(np.mean(s.det_fluence))
            det_fluence_std.append(np.std(s.det_fluence))

            inj_width.append(np.mean(s.width))
            inj_snr.append(np.mean(s.snr))

            noise_std.append(np.mean(s.noise_std))
            det_amp.append(np.mean(s.det_amp))
            det_amp_std.append(np.std(s.det_amp))

        noise_std = np.array(noise_std)
        det_snr = np.array(det_snr)
        det_width = np.array(det_width)
        det_amp = np.array(det_amp)
        det_width_std = np.array(det_width_std)
        inj_snr = np.array(inj_snr)
        inj_width = np.array(inj_width)

        #calculate inj_fluence from snr and width
        inj_fluence = inj_snr*inj_width/0.3989
        det_fluence = np.array(det_fluence)
        det_fluence_std = np.array(det_fluence_std)
        det_snr_std = np.array(det_snr_std)
        unique_snrs, unique_widths, det_matrix_width = create_matrix(inj_snr, inj_width, det_width,norm=1)
        unique_snrs, unique_widths, det_matrix_snr = create_matrix(inj_snr, inj_width, det_snr,norm=0)
        unique_snrs, unique_widths, det_matrix_width_std = create_matrix(inj_snr, inj_width, det_width_std,norm=-1)
        unique_snrs, unique_widths, det_matrix_snr_std = create_matrix(inj_snr, inj_width, det_snr_std,norm=-1)
        #convert to ms
        unique_widths = unique_widths*1000
        fig,axes = plt.subplots(2,2,figsize=(10,10))
        #set maximum and minumum colors to 0.5 and 1.5
        mesh = axes[0,0].pcolormesh(unique_widths, unique_snrs, det_matrix_width)
        mesh.set_clim(0.8,1.2)
        cbar = plt.colorbar(mesh)
        cbar.set_label("Detected Width / Injected Width")
        axes[0,0].set_xlabel("Injected Width (ms)")
        axes[0,0].set_ylabel("Injected SNR")
        axes[0,0].set_title("Detected Width")

        mesh = axes[0,1].pcolormesh(unique_widths, unique_snrs, det_matrix_snr)
        mesh.set_clim(0.5,1.5)
        cbar = plt.colorbar(mesh)
        cbar.set_label("Detected SNR / Injected SNR")
        axes[0,1].set_xlabel("Injected Width (ms)")
        axes[0,1].set_ylabel("Injected SNR")
        axes[0,1].set_title("Detected SNR")

        mesh = axes[1,0].pcolormesh(unique_widths, unique_snrs, det_matrix_width_std*1000)
        mesh.set_clim(0,10)
        cbar = plt.colorbar(mesh)
        cbar.set_label("Detected Width STD (ms)")
        axes[1,0].set_xlabel("Injected Width (ms)")
        axes[1,0].set_ylabel("Injected SNR")
        axes[1,0].set_title("Detected Width STD")

        mesh = axes[1,1].pcolormesh(unique_widths, unique_snrs, det_matrix_snr_std)
        mesh.set_clim(0,1)
        cbar = plt.colorbar(mesh)
        cbar.set_label("Detected SNR STD")
        axes[1,1].set_xlabel("Injected Width (ms)")
        axes[1,1].set_ylabel("Injected SNR")
        axes[1,1].set_title("Detected SNR STD")
        plt.tight_layout()
        if hasattr(self, "base_fn"):
            plt.savefig(self.base_fn + "_snr_amp.png")
        else:
            plt.show()
        unique_fluences, unique_width_fs, det_matrix_width_f = create_matrix(inj_fluence, inj_width, det_width,norm=1)
        unique_fluences, unique_width_fs, det_matrix_fluence = create_matrix(inj_fluence, inj_width, det_fluence,norm=0)
        unique_fluences, unique_width_fs, det_matrix_width_f_std = create_matrix(inj_fluence, inj_width, det_width_std,norm=-1)
        unique_fluences, unique_width_fs, det_matrix_fluence_std = create_matrix(inj_fluence, inj_width, det_fluence_std,norm=-1)
        #convert to ms
        unique_width_fs = unique_width_fs*1000
        fig,axes = plt.subplots(2,2,figsize=(10,10))
        #set maximum and minumum colors to 0.5 and 1.5
        mesh = axes[0,0].pcolormesh(unique_width_fs, unique_fluences, det_matrix_width_f)
        mesh.set_clim(0.8,1.2)
        cbar = plt.colorbar(mesh)
        cbar.set_label("Detected Width_F / Injected Width_F")
        axes[0,0].set_xlabel("Injected Width_F (ms)")
        axes[0,0].set_ylabel("Injected FLUENCE")
        axes[0,0].set_title("Detected Width_F")

        mesh = axes[0,1].pcolormesh(unique_width_fs, unique_fluences, det_matrix_fluence)
        mesh.set_clim(0.8,1.2)
        cbar = plt.colorbar(mesh)
        cbar.set_label("Detected FLUENCE / Injected FLUENCE")
        axes[0,1].set_xlabel("Injected Width_F (ms)")
        axes[0,1].set_ylabel("Injected FLUENCE")
        axes[0,1].set_title("Detected FLUENCE")

        mesh = axes[1,0].pcolormesh(unique_width_fs, unique_fluences, det_matrix_width_f_std*1000)
        mesh.set_clim(0,10)
        cbar = plt.colorbar(mesh)
        cbar.set_label("Detected Width_F STD (ms)")
        axes[1,0].set_xlabel("Injected Width_F (ms)")
        axes[1,0].set_ylabel("Injected FLUENCE")
        axes[1,0].set_title("Detected Width_F STD")

        mesh = axes[1,1].pcolormesh(unique_width_fs, unique_fluences, det_matrix_fluence_std)
        # mesh.set_clim(0,1)
        cbar = plt.colorbar(mesh)
        cbar.set_label("Detected FLUENCE STD")
        axes[1,1].set_xlabel("Injected Width_F (ms)")
        axes[1,1].set_ylabel("Injected FLUENCE")
        axes[1,1].set_title("Detected FLUENCE STD")
        plt.tight_layout()
        if hasattr(self, "base_fn"):
            plt.savefig(self.base_fn + "_fluence_amp.png")
        else:
            plt.show()
        plt.close()
        ind = np.argsort(inj_snr)
        self.det_snr = det_snr[ind]
        self.det_snr_std = det_snr_std[ind]
        self.inj_snr = inj_snr[ind]
        self.det_width = det_width[ind]
        self.det_width_std = det_width_std[ind]
        self.inj_width = inj_width[ind]
        self.det_fluence = det_fluence[ind]
        self.det_fluence_std = det_fluence_std[ind]
        self.inj_fluence = inj_fluence[ind]

        self.sorted_inject = np.array(self.sorted_inject)[ind]
        # take the average of the last 3 for the error
        #hardcode larger than widt 10ms
        sm, wm = np.meshgrid(unique_snrs, unique_widths, indexing="ij")
        mask = (wm > 12) & (wm < 18) & (sm > 3)

        sfm, wfm = np.meshgrid(unique_fluences, unique_width_fs, indexing="ij")
        fmask = (wfm > 12) & (wfm < 18) & (sfm > 0.14)
        #add the errors in quadrature
        self.detect_error_snr = np.sqrt(np.nanmean(det_matrix_snr_std[mask]**2))
        self.detect_error_width = np.sqrt(np.nanmean(det_matrix_width_std[mask]**2))
        self.detect_error_fluence = np.sqrt(np.nanmean(det_matrix_fluence_std[fmask]**2))
        print(f"SNR error: {self.detect_error_snr}, width error: {self.detect_error_width}, fluence error: {self.detect_error_fluence}")

    def detected_truth(self, si, truth_arr):
        # if we have detected truth array then or the thing, if not then create
        if hasattr(si, "detected"):
            si.detected = si.detected | truth_arr
        else:
            si.detected = truth_arr

    def compare(self, fn, plot=True, title="detection_curve"):
        from read_positive_burst import read_positive_burst_inj

        matched = np.zeros(len(self.dm_arr))
        time_tol = 0.5
        dm_tol = 10
        snr_tol = 1e-3
        width_tol = 1e-4
        for i,csv in enumerate(fn):
            (
                dm,
                burst_time,
                boxcar_det_snr,
                inj_snr,
                inj_width,
                MJD,
            ) = read_positive_burst_inj(csv)
            before_num = [sum(si.detected) for si in self.sorted_inject]
            for t, d, inj_s, inj_w in zip(burst_time, dm, inj_snr, inj_width):
                # here we gotta match the values
                t_low = t - time_tol
                t_hi = t + time_tol
                dm_low = d - dm_tol
                dm_hi = d + dm_tol
                # print(t_low,t_hi,dm_low,dm_hi)
                for si in self.sorted_inject:

                    t_snr = (np.mean(si.snr) > (inj_s - snr_tol)) & (
                        np.mean(si.snr) < (inj_s + snr_tol)
                    )
                    t_width = (np.mean(si.width) > (inj_w - width_tol)) & (
                        np.mean(si.width) < (inj_w + width_tol)
                    )
                    if t_snr&t_width:
                        dm_arr = si.dm
                        t_arr = si.toas
                        truth_dm = (dm_arr < dm_hi) & (dm_arr > dm_low)
                        truth_t = (t_arr < t_hi) & (t_arr > t_low)
                        total_truth = truth_dm & truth_t
                        self.detected_truth(si, total_truth)
            after_num = [sum(si.detected) for si in self.sorted_inject]
            extra = np.array(after_num) - np.array(before_num)
            if i>0:
                print(csv,sum(extra))
        det_frac = []

        #remove the lowest snr so that our statistics don't get ruined
        mask = (self.inj_snr > 1)#&(self.inj_width>5e-3)
        inj_snr = self.inj_snr[mask]
        inj_width = self.inj_width[mask]
        inj_fluence = self.inj_fluence[mask]
        sorted_inject = self.sorted_inject[mask]


        for si in sorted_inject:
            # print(f"snr {np.mean(si.snr)}",f"width {np.mean(si.width)}",'detection frac',sum(si.detected)/len(si.detected))
            det_frac.append(sum(si.detected)/len(si.detected))
        det_frac = np.array(det_frac)
        # get lists to plot
        unique_snrs, unique_widths, det_frac_matrix_snr = create_matrix(inj_snr, inj_width, det_frac,norm=2)
        unique_fluence, unique_width_fs, det_matrix_fluence = create_matrix(inj_fluence, inj_width, det_frac,norm=2)

        # detected_sample = sorted_inject[(self.inj_snr>2)&(self.inj_width>10e-3)]
        detected_sample = sorted_inject

        detected_amplitudes_snr = list(s.det_snr[s.detected] for s in detected_sample)
        detected_amplitudes_fluence = list(s.det_fluence[s.detected] for s in detected_sample)
        detected_widths = list(s.det_std[s.detected] for s in detected_sample)
        all_det_amplitudes_snr = list(s.det_snr for s in detected_sample)
        all_det_amplitudes_fluence = list(s.det_fluence for s in detected_sample)
        all_det_widths = list(s.det_std for s in detected_sample)
        #flatten the lists
        detected_amplitudes_snr = [item for sublist in detected_amplitudes_snr for item in sublist]
        detected_amplitudes_fluence = [item for sublist in detected_amplitudes_fluence for item in sublist]
        detected_widths = [item for sublist in detected_widths for item in sublist]
        all_det_amplitudes_snr = [item for sublist in all_det_amplitudes_snr for item in sublist]
        all_det_amplitudes_fluence = [item for sublist in all_det_amplitudes_fluence for item in sublist]
        all_det_widths = [item for sublist in all_det_widths for item in sublist]

        self.all_det_amplitudes_snr = np.array(all_det_amplitudes_snr)
        self.all_det_amplitudes_fluence = np.array(all_det_amplitudes_fluence)
        self.all_det_widths = np.array(all_det_widths)
        self.detected_widths = np.array(detected_widths)
        self.detected_amplitudes_snr = np.array(detected_amplitudes_snr)
        self.detected_amplitudes_fluence = np.array(detected_amplitudes_fluence)

        self.unique_snrs = unique_snrs
        self.unique_widths = unique_widths
        self.det_frac_matrix_snr = det_frac_matrix_snr
        self.unique_fluence = unique_fluence
        self.unique_width_fs = unique_width_fs
        self.det_matrix_fluence = det_matrix_fluence
        num_bins = 20
        self.bin_detections_2d(self.all_det_amplitudes_snr, self.detected_amplitudes_snr, self.all_det_widths, self.detected_widths, num_bins=num_bins,plot=False, fluence=False)
        self.bin_detections_2d(self.all_det_amplitudes_fluence, self.detected_amplitudes_fluence, self.all_det_widths, self.detected_widths, num_bins=num_bins,plot=False, fluence=True)
        fig,axes = plt.subplots(2,2,figsize=(10,10))
        mesh = axes[0,0].pcolormesh(unique_widths*1000, unique_snrs, det_frac_matrix_snr)
        mesh.set_clim(0,1)
        cbar = plt.colorbar(mesh)
        cbar.set_label("Detection Fraction")
        axes[0,0].set_xlabel("Width (ms)")
        axes[0,0].set_ylabel("S/N")

        mesh = axes[0,1].pcolormesh(self.detected_bin_midpoints_snr[1]*1000,self.detected_bin_midpoints_snr[0],self.detected_det_frac_snr)
        cbar = plt.colorbar(mesh)
        mesh.set_clim(0,1)
        cbar.set_label("Detection Fraction (binned)")
        axes[1,0].set_xlabel("Width (ms)")
        axes[1,0].set_ylabel("S/N")
        plt.tight_layout()

        mesh = axes[1,0].pcolormesh(unique_width_fs*1000, unique_fluence, det_matrix_fluence)
        mesh.set_clim(0,1)
        cbar = plt.colorbar(mesh)
        cbar.set_label("Detection Fraction")
        axes[1,0].set_xlabel("Width (ms)")
        axes[1,0].set_ylabel("Fluence (Jy s)")

        mesh = axes[1,1].pcolormesh(self.detected_bin_midpoints_fluence[1]*1000,self.detected_bin_midpoints_fluence[0],self.detected_det_frac_fluence)
        cbar = plt.colorbar(mesh)
        mesh.set_clim(0,1)
        cbar.set_label("Detection Fraction (binned)")
        axes[1,1].set_xlabel("Width (ms)")
        axes[1,1].set_ylabel("Fluence (Jy s)")
        plt.tight_layout()


        plt.savefig(f"{title}.png")
        plt.close()
        # plt.show()


        #high width sample
        # unique_widths = np.unique(self.inj_width)
        # for uw in unique_widths:
        #     detected_sample = self.sorted_inject[(self.inj_width==uw)&(self.inj_snr>1)]
        #     true_inj_amplitude = list(np.mean(s.snr) for s in detected_sample)
        #     true_inj_ampltiude_all = np.array(list(s.snr for s in detected_sample))
        #     #flatten the list
        #     true_inj_ampltiude_all = [item for sublist in true_inj_ampltiude_all for item in sublist]

        #     detected_frac = list(sum(s.detected)/len(s.detected) for s in detected_sample)

        #     detected_amplitudes = list(s.det_snr[s.detected] for s in detected_sample)
        #     detected_widths = list(s.det_std[s.detected] for s in detected_sample)
        #     all_det_amplitudes = list(s.det_snr for s in detected_sample)
        #     all_det_widths = list(s.det_std for s in detected_sample)
        #     #flatten the lists
        #     detected_amplitudes = np.array([item for sublist in detected_amplitudes for item in sublist])
        #     detected_widths = np.array([item for sublist in detected_widths for item in sublist])
        #     all_det_amplitudes = np.array([item for sublist in all_det_amplitudes for item in sublist])
        #     all_det_widths = np.array([item for sublist in all_det_widths for item in sublist])


        #     self.bin_detections(all_det_amplitudes, detected_amplitudes)
        #     fig,axes = plt.subplots(1,2,figsize=(10,10))
        #     axes[0].plot(true_inj_amplitude,detected_frac)
        #     axes[0].set_xlabel("True S/N")
        #     axes[0].set_ylabel("Detection Fraction")
        #     axes[0].plot(self.detected_bin_midpoints,self.detected_det_frac)
        #     sc = axes[1].scatter(np.arange(len(all_det_amplitudes)),all_det_amplitudes/true_inj_ampltiude_all,c=true_inj_ampltiude_all)
        #     cbar = plt.colorbar(sc)
        #     cbar.set_label("True S/N")
        #     axes[1].set_xlabel("Detection Number")
        #     axes[1].set_ylabel("Detected S/N/True S/N")
        #     plt.title(f"Width {uw*1000} ms")
        #     plt.tight_layout()
        #     plt.savefig(f"width_{uw*1000}_ms.png")

        # unique_snrs = np.unique(self.inj_snr)
        # for us in unique_snrs:
        #     if us<1:
        #         continue
        #     detected_sample = self.sorted_inject[self.inj_snr==us]
        #     true_inj_width = np.array(list(np.mean(s.width) for s in detected_sample))
        #     true_inj_width_all = np.array(list(s.width for s in detected_sample))
        #     #flatten the list
        #     true_inj_width_all = np.array([item for sublist in true_inj_width_all for item in sublist])
        #     detected_frac = list(sum(s.detected)/len(s.detected) for s in detected_sample)
        #     detected_amplitudes = list(s.det_snr[s.detected] for s in detected_sample)
        #     detected_widths = list(s.det_std[s.detected] for s in detected_sample)
        #     all_det_amplitudes = list(s.det_snr for s in detected_sample)
        #     all_det_widths = list(s.det_std for s in detected_sample)
        #     #flatten the lists
        #     detected_amplitudes = np.array([item for sublist in detected_amplitudes for item in sublist])
        #     detected_widths = np.array([item for sublist in detected_widths for item in sublist])
        #     all_det_amplitudes = np.array([item for sublist in all_det_amplitudes for item in sublist])
        #     all_det_widths = np.array([item for sublist in all_det_widths for item in sublist])


        #     self.bin_detections(all_det_widths, detected_widths)
        #     fig,axes = plt.subplots(1,2,figsize=(10,10))
        #     axes[0].scatter(true_inj_width*1e3,detected_frac)
        #     axes[0].set_xlabel("True width")
        #     axes[0].set_ylabel("Detection Fraction")
        #     axes[0].scatter(self.detected_bin_midpoints*1e3,self.detected_det_frac)
        #     sc = axes[1].scatter(np.arange(len(all_det_widths)),all_det_widths/true_inj_width_all,c=true_inj_width_all*1e3)
        #     #set colorbar
        #     cbar = plt.colorbar(sc)
        #     cbar.set_label("True width (ms)")
        #     axes[1].set_xlabel("index")
        #     axes[1].set_ylabel("Detected width / True width")

        #     plt.title(f"S/N {us}")
        #     plt.tight_layout()
        #     plt.savefig(f"snr_{us}_ms.png")




    def bin_detections(self,all_det_vals, detected_det_vals, num_bins=20,plot=False):
        # set the number of data points per bin
        self.num_points_per_bin = len(all_det_vals) // num_bins
        print("number of points in each bin ", self.num_points_per_bin)

        # calculate the bin edges based on the percentiles of the data
        bin_edges = np.quantile(all_det_vals, np.linspace(0, 1, num_bins+1))

        # use the digitize function to assign data points to bins
        bin_assignments_all = np.digitize(all_det_vals, bin_edges)
        bin_assignments_detected = np.digitize(detected_det_vals, bin_edges)

        #count the number in each bin
        hist_all, _ = np.histogram(all_det_vals, bins=bin_edges)
        hist_detected, _ = np.histogram(detected_det_vals, bins=bin_edges)

        detected_det_frac = hist_detected/hist_all
        bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2

        #remove nans in detected_det_frac
        bin_midpoints = bin_midpoints[~np.isnan(detected_det_frac)]
        hist_all = hist_all[~np.isnan(detected_det_frac)]
        hist_detected = hist_detected[~np.isnan(detected_det_frac)]
        detected_det_frac = detected_det_frac[~np.isnan(detected_det_frac)]
        if plot:
            fig,axes = plt.subplots(1,2)
            axes[0].scatter(bin_midpoints,detected_det_frac,label="P(Detected|amplitude_detected)")
            axes[1].scatter(bin_midpoints,hist_detected,label="the detected pulses")
            axes[1].scatter(bin_midpoints,hist_all,alpha=0.5,label="all the pulses")
            axes[1].legend()
        self.detected_bin_midpoints = np.array(bin_midpoints)
        self.detected_det_frac = detected_det_frac
        #remove the last 2 bins
        self.detected_bin_midpoints = self.detected_bin_midpoints[:-2]
        self.detected_det_frac = self.detected_det_frac[:-2]

    def bin_detections_2d(self,all_det_vals, detected_det_vals, all_width_vals, detected_width_vals, num_bins=20,plot=False, fluence=False):
        # set the number of data points per bin
        # if fluence is set to true, all the "snr" things are supposed to be fluence
        # self.num_points_per_bin_snr = len(all_det_vals) // num_bins
        # self.num_points_per_bin_width = len(all_width_vals) // num_bins
        # print("number of points in each bin amplitude", self.num_points_per_bin_snr)
        # print("number of points in each bin width", self.num_points_per_bin_width)

        # calculate the bin edges based on the percentiles of the data

        all_mask = (all_det_vals<(max(self.inj_snr)+0.5)) & (all_width_vals<(max(self.inj_width)+1e-3)) & (all_width_vals>2e-3)
        det_mask = (detected_det_vals<(max(self.inj_snr)+0.5)) & (detected_width_vals<(max(self.inj_width)+1e-3))& (detected_width_vals>2e-3)
        all_det_vals = all_det_vals[all_mask]
        detected_det_vals = detected_det_vals[det_mask]
        all_width_vals = all_width_vals[all_mask]
        detected_width_vals = detected_width_vals[det_mask]

        bin_edges_snr = np.quantile(all_det_vals, np.linspace(0, 1, num_bins+1))
        bin_edges_width = np.quantile(all_width_vals, np.linspace(0, 1, num_bins+1))
        # bin_edges_snr = np.histogram_bin_edges(all_det_vals, bins=num_bins)
        # bin_edges_width = np.histogram_bin_edges(all_width_vals, bins=num_bins)
        #remove bin edges above max of self.inj_snr
        # if fluence:
        #     bin_edges_snr = bin_edges_snr[bin_edges_snr<max(self.inj_fluence)+0.4]
        # else:
        #     bin_edges_snr = bin_edges_snr[bin_edges_snr<max(self.inj_snr)+0.5]
        # bin_edges_width = bin_edges_width[bin_edges_width<max(self.inj_width)+5e-3]

        all_2d_hist, xedges, yedges = np.histogram2d(all_det_vals, all_width_vals, bins=(bin_edges_snr,bin_edges_width))
        detected_2d_hist, xedges, yedges = np.histogram2d(detected_det_vals, detected_width_vals, bins=(bin_edges_snr,bin_edges_width))
        detected_2d_frac = detected_2d_hist/all_2d_hist
        #plot detected 2d frac
        if plot:
            fig,axes = plt.subplots(1,1,figsize=(10,10))
            mesh = axes.pcolormesh(xedges, yedges, detected_2d_frac)
            cbar = plt.colorbar(mesh)
            cbar.set_label("Detection Fraction")
            if fluence:
                axes.set_xlabel("fluence")
            else:
                axes.set_xlabel("S/N")
            axes.set_ylabel("Width (ms)")
            plt.show()

        #find bin midpoints
        x_bin_midpoints = (xedges[1:] + xedges[:-1])/2
        y_bin_midpoints = (yedges[1:] + yedges[:-1])/2

        if fluence:
            self.detected_bin_midpoints_fluence = (x_bin_midpoints,y_bin_midpoints)
            self.detected_det_frac_fluence = detected_2d_frac
        else:
            self.detected_bin_midpoints_snr = (x_bin_midpoints,y_bin_midpoints)
            self.detected_det_frac_snr = detected_2d_frac

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", action="store_false", default=True, help="Set to do inj_stats analysis"
    )
    parser.add_argument(
        "-ds", default=3, type=int, help="Downsample when doing a gaussian fit"
    )
    parser.add_argument(
        "-l",
        nargs="+",
        help="list of filterbank files or positive burst csv files",
        required=True,
    )
    parser.add_argument(
        "-multi",
        help="enable multiprocessing with 10 cores",
        action="store_true",
    )

    args = parser.parse_args()
    do_fluence_calc = args.d
    downsamp = args.ds
    if do_fluence_calc:
        inj_samples = "sample_injections.npz"
        filfiles = args.l
        init = {"filfiles": filfiles, "inj_samp": inj_samples, "downsamp": downsamp}
        inj_stats = inject_stats(**init)
        inj_stats.load_inj_samp()
        inj_stats.match_inj()
        print(len(inj_stats.toa_arr))
        inj_stats.calculate_snr(args.multi)
        with open("inj_stats.dill", "wb") as of:
            dill.dump(inj_stats, of)
    else:
        with open("inj_stats.dill", "rb") as inf:
            inj_stats = dill.load(inf)
        fns = args.l
        inj_stats = inject_stats(**inj_stats.__dict__)
        inj_stats.repopulate_io()
        inj_stats.amplitude_statistics()
        inj_stats.compare(fns)
        with open("inj_stats_fitted.dill", "wb") as of:
            dill.dump(inj_stats, of)
