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
    # mask the data but set to the mean of the channel
    mask_vals = np.median(data, axis=1)
    for i in range(len(mask_vals)):
        _ = data[i, :]
        _m = mask[i, :]
        _[_m] = mask_vals[i]
        data[i, :] = _
    return data, masked_chans


def extract_plot_data(data,masked_chans,dm,downsamp,nsamps_start_zoom,nsamps_end_zoom):
    # make a copy to plot the waterfall
    waterfall_dat = copy.deepcopy(data)
    waterfall_dat = waterfall_dat.downsample(tfactor=downsamp)
    dat_ts = np.mean(waterfall_dat[~masked_chans, :], axis=0)
    dat_ts = dat_ts[int(nsamps_start_zoom / downsamp) : int(nsamps_end_zoom / downsamp)]

    waterfall_dat = waterfall_dat.downsample(ffactor=8)
    waterfall_dat = waterfall_dat.normalise()
    waterfall_dat = waterfall_dat[
        :, int(nsamps_start_zoom / downsamp) : int(nsamps_end_zoom / downsamp)
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
        print(sum(masked_chans))
    data = data.dedisperse(dm)
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

        while amp==-1:
            #fit has failed, get a larger time window and try again
            # make a copy to plot the waterfall
            t_start = t_start - 1
            t_dur = t_dur + 2
            print(t_start,t_dur)
            if (t_start<0)|((t_start+t_dur)>(te-ts)):
                break
            nsamps_start_zoom = int(t_start / tsamp)
            nsamps_end_zoom = int((t_dur+t_start) / tsamp)
            print(nsamps_start_zoom,nsamps_end_zoom)
            waterfall_dat, dat_ts = extract_plot_data(data,masked_chans,dm,downsamp,nsamps_start_zoom,nsamps_end_zoom)
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
            amp, std, loc, sigma_width = fit_SNR_manual(
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
            #scale the ts with the std so everythin is in units of noise
            #FLUENCE = fit_FLUENCE(
            #    ts_no_ds/std,
            #    tsamp,
            #    3 * sigma_width,
            #    nsamp=int(loc / tsamp),
            #    ds_data=waterfall_dat,
            #    plot=False,
            #)
        else:
            FLUENCE = -1
    else:
        # fit using downsampled values
        # this is mostly used for the injections
        try:
            amp, std, loc, sigma_width = autofit_pulse(
                dat_ts,
                tsamp * downsamp,
                fit_del,
                nsamps=int(t_dur / 2 / tsamp / downsamp),
                ds_data=waterfall_dat,
                downsamp=downsamp,
                plot=False,
                plot_name=plot_name,
                width = guess_width
            )
            #refit with new initial params
            amp, std, loc, sigma_width = autofit_pulse(
                dat_ts,
                tsamp * downsamp,
                fit_del,
                nsamps=int(loc / tsamp / downsamp),
                ds_data=waterfall_dat,
                downsamp=downsamp,
                plot=True,
                plot_name=plot_name,
                width = std
            )
        except Exception as e:
            print(e)
            amp, std, loc, sigma_width = -1, -1, -1, -1
        #scale std by the sqrt of non masked chans
        std = std * np.sqrt(sum(~masked_chans)/len(masked_chans))
        SNR = amp / std
        # because loc is predetermined set start and end a predifined spot
        ts_no_ds_zoom_start = int(4.1/tsamp)
        ts_no_ds_zoom_end = int(5.9/tsamp)
        ts_no_ds = data[:, ts_no_ds_zoom_start : ts_no_ds_zoom_end]
        ts_no_ds = np.mean(ts_no_ds[~masked_chans, :], axis=0)
        #FLUENCE = fit_FLUENCE(
        #    ts_no_ds/std,
        #    tsamp,
        #    fit_del,
        #    nsamp=int(loc / tsamp),
        #    ds_data=waterfall_dat,
        #    plot=False,
        #)
    FLUENCE = -1
    # recalculate the amplitude given a gaussian pulse shape
    gaussian_amp = FLUENCE / sigma_width / np.sqrt(2 * np.pi)
    print("filename:", gf, "downsample:", downsamp, "FLUENCE:", FLUENCE)
    approximate_toa = g.header.tstart + ((te+ts)/2)/86400
    return FLUENCE, std, amp, gaussian_amp, sigma_width, SNR, approximate_toa

def find_polynomial_fit(x_std, ts_std):
    rchi2 = 100
    i = 1
    rchi2_arr = []
    poly_arr = []
    coeffs_arr = []
    for i in range(10):
        coeffs = np.polyfit(x_std, ts_std, i)
        poly = np.poly1d(coeffs)
        # Calculate the reduced chi2 of the fit
        rchi2 = np.sum((ts_std - poly(x_std)) ** 2 / (np.std(ts_std[1:500]) ** 2)) / (len(ts_std) - i)
        print("rchi2", rchi2, "i", i)
        rchi2_arr.append(rchi2)
        poly_arr.append(poly)
        coeffs_arr.append(coeffs)
    # find the minimum rchi2
    rchi2_arr = (np.array(rchi2_arr)-1)**2
    ind = np.argmin(rchi2_arr)
    poly = poly_arr[ind]
    coeffs = coeffs_arr[ind]
    # coeffs = np.polyfit(x_std, ts_std, 10)
    # poly = np.poly1d(coeffs)

    return poly, coeffs

def autofit_pulse(ts, tsamp, width, nsamps, ds_data, downsamp, plot=True, plot_name="", width = 0.01):
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
    xind = np.array(list(range(len(ts_sub)))) * tsamp

    max_l = minimize(
        log_likelihood,
        [mamplitude, max_time, width, 0],
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
    print(f"Fitted loc:{loc} amp:{Amplitude} std:{std} width sigma:{sigma_width}")
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

    return Amplitude, std, loc, sigma_width

def fit_FLUENCE(ts, tsamp, width, nsamp, ds_data, plot=False):
    # calculates the FLUENCE given a timeseries
    w_bin = int(width / tsamp)
    x = np.linspace(0, tsamp * len(ts), len(ts))
    if (nsamp+w_bin)>ts.shape[0]:
        #if pulse is on very edge then just reset nsamp to the middle, unfortunately, can't do anything about that
        nsamp = int(ts.shape[0]/2)
    ts_std = np.delete(ts, range(int(nsamp - w_bin), int(nsamp + w_bin)))
    x_std = np.delete(x, range(int(nsamp - w_bin), int(nsamp + w_bin)))
    # fit a polynomial
    coeffs = np.polyfit(x_std, ts_std, 10)
    poly = np.poly1d(coeffs)
    # subtract the mean
    ts_sub = ts - poly(x)
    ts_start = ts[: int(nsamp - w_bin)]
    x_start = x[: int(nsamp - w_bin)]
    ts_start = ts_start - poly(x_start)
    ts_end = ts[int(nsamp + w_bin) :]
    x_end = x[int(nsamp + w_bin) :]
    ts_end = ts_end - poly(x_end)

    fluence_noise = np.trapz(ts_start, x_start) + np.trapz(ts_end, x_end)
    print("fluence noise", fluence_noise)
    # grab just the window
    ts_fluence = ts_sub[nsamp - w_bin : nsamp + w_bin]
    x_fluence = x[nsamp - w_bin : nsamp + w_bin]
    # fluence = np.trapz(ts_fluence,x_fluence)
    # just integrate over the whole thing for fluence
    fluence = np.trapz(ts_sub, x)
    if plot:
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].plot(x_start, ts_start)
        axs[0, 0].plot(x_end, ts_end)
        axs[0, 0].set_title("burst_removed")
        axs[1, 0].plot(x, ts_sub)
        axs[1, 0].set_title("baseline subtracted")
        axs[1, 1].plot(x, ts)
        axs[1, 1].set_title("OG time series")
        plt.show()

    return fluence


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
    std = np.std(ts_std_sub)
    # remove rms
    # fit this to a gaussian using ML
    mamplitude = np.max(ts_sub)
    max_time = nsamps * tsamp
    # x axis of the fit

    xind = np.array(list(range(len(ts_sub)))) * tsamp
    # print("init values",[mamplitude, max_time, width * 0.1, np.mean(ts_sub)])
    max_l = minimize(
        log_likelihood,
        [mamplitude, max_time, 1e-2, np.mean(ts_sub)],
        args=(xind, ts_sub, std),
        bounds=((1e-4, None), (0, max(x)), (1e-6, 5e-2), (-2, 2)),
        method="Nelder-Mead",
    )
    fitx = max_l.x
    print(fitx)
    # double the resolution
    xind_fit = np.linspace(min(xind), max(xind), len(xind) * 2)
    y_fit = gaussian(xind_fit, fitx[0], fitx[1], fitx[2], fitx[3])
    fig,axes = plt.subplots(1,3,figsize=(10,10))
    cmap = plt.get_cmap("magma")
    axes[0].imshow(ds_data, aspect="auto", cmap=cmap)
    #plot the polyfit
    axes[2].plot(x, np.interp(x,x_std,poly(x_std)), lw=5, alpha=0.7)
    axes[2].scatter(x_std,ts_std,alpha=0.5)
    axes[2].scatter(x,ts,alpha=0.5)

    k = axes[1].plot(xind, ts_sub)
    (my_plot,) = axes[1].plot(xind_fit, y_fit, lw=5,alpha=0.7)
    # ax.margins(x=0)
    axcolor = "lightgoldenrodyellow"
    pl_ax = plt.axes([0.1, 0.05, 0.78, 0.03], facecolor=axcolor)
    p_ax = plt.axes([0.1, 0.1, 0.78, 0.03], facecolor=axcolor)
    w_ax = plt.axes([0.1, 0.15, 0.78, 0.03], facecolor=axcolor)
    pl = Slider(pl_ax, "peak loc", 0.0, np.max(x), valinit=fitx[1], valstep=1e-3)
    p = Slider(p_ax, "peak", 0.0, 1, valinit=fitx[0], valstep=1e-5)
    w = Slider(w_ax, "width", 0.0, 0.05, valinit=fitx[2], valstep=1e-3)
    but_ax = plt.axes([0.1, 0.02, 0.3, 0.03], facecolor=axcolor)
    but_save = plt.axes([0.5, 0.02, 0.3, 0.03], facecolor=axcolor)

    skipb = Button(but_ax, "Skip")
    saveb = Button(but_save, "Save")

    # plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    global x_new
    x_new = [-1, -1, -1, -1]

    def update(val):
        peak_loc = pl.val
        peak = p.val
        sigma = w.val
        a = np.mean(ts_sub)
        # refit with new values
        max_l = minimize(
            log_likelihood,
            [peak, peak_loc, sigma, a],
            bounds=((1e-4, None), (0, max(x)), (1e-6, 5e-2), (-2, 2)),
            args=(xind, ts_sub, std),
            method="Nelder-Mead",
        )
        for i, v in enumerate(max_l.x):
            x_new[i] = v

        print("new fit: ", x_new)
        new_fit = gaussian(xind_fit, x_new[0], x_new[1], x_new[2], x_new[3])
        my_plot.set_ydata(new_fit)
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

    SNR = Amplitude / std
    # once we have calculated the location
    print(f"Fitted loc:{loc} amp:{Amplitude} std:{std} width sigma:{sigma_width}")
    return Amplitude, std, loc, sigma_width


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
            self.processed = False
        else:
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
        self.base_fn = "_".join(splits)

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
            snr_str = sp_[-1]
            snr_str = snr_str.strip(".fil").strip("SNR")
            snr = np.round(float(snr_str), 4)
            snr_ind = np.round(self.snr_arr, 4) == snr
            cur_snr = self.snr_arr[snr_ind]
            cur_toa = self.toa_arr[snr_ind]
            cur_dm = self.dm_arr[snr_ind]
            cur_width = self.width_arr[snr_ind]
            if len(cur_snr)==0:
                print("WARNING NO MATCHING SNR, THIS SHOULD NOT HAPPEN NORMALLY. IF YOU ARE NOT EXPECTING THIS WARNING THEN CHECK THE FILES CREATED")
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

    def amplitude_statistics(self):
        det_snr = []
        inj_snr = []
        det_snr_std = []
        noise_std = []
        det_amp = []
        det_amp_std = []
        for s in self.sorted_inject:
            det_snr.append(np.mean(s.det_snr))
            det_snr_std.append(np.std(s.det_snr))
            inj_snr.append(np.mean(s.snr))
            noise_std.append(np.mean(s.noise_std))
            det_amp.append(np.mean(s.det_amp))
            det_amp_std.append(np.std(s.det_amp))

        noise_std = np.array(noise_std)
        det_snr = np.array(det_snr)
        inj_snr = np.array(inj_snr)
        det_snr_std = np.array(det_snr_std)
        try:
            p_snr = np.polyfit(det_snr[1:], inj_snr[1:], deg=1)
        except:
            p_snr = np.polyfit(det_snr, inj_snr, deg=1)
        poly_snr = np.poly1d(p_snr)


        fig, axes = plt.subplots(1, 3)
        x = np.linspace(0,10)
        axes[0].plot(x, poly_snr(x))
        axes[0].errorbar(det_snr, inj_snr, xerr=np.array(det_snr_std), fmt=".")
        axes[0].set_ylabel("Injected SNR")
        axes[0].set_xlabel("Detected SNR")
        axes[1].scatter(inj_snr,noise_std)
        axes[1].set_xlabel("Injected SNR")
        axes[1].set_ylabel("Detected noise")
        axes[2].errorbar(inj_snr, det_amp, yerr=np.array(det_amp_std), fmt=".")
        if hasattr(self, "base_fn"):
            plt.savefig(self.base_fn + "_amp.png")
        else:
            plt.show()
        plt.close()
        ind = np.argsort(inj_snr)
        self.det_snr = det_snr[ind]
        self.det_snr_std = det_snr_std[ind]
        self.inj_snr = inj_snr[ind]
        self.poly_snr = p_snr
        # take the average of the last 3 for the error
        self.detect_error_snr = np.sqrt(np.mean(self.det_snr_std[-3:]**2))
        print(self.detect_error_snr)

    def calculate_fluence_statistics(self):
        det_fluence = []
        det_fluence_std = []
        inj_fluence = []
        for s in self.sorted_inject:
            det_fluence.append(np.mean(s.det_fluence))
            det_fluence_std.append(np.std(s.det_fluence))
            inj_fluence.append(np.mean(s.fluence))
        det_fluence = np.array(det_fluence)
        inj_fluence = np.array(inj_fluence)
        det_fluence_std = np.array(det_fluence_std)
        p = np.polyfit(det_fluence, inj_fluence, deg=1)
        x = np.linspace(0, 0.004)
        poly = np.poly1d(p)

        plt.figure()
        plt.plot(x, poly(x))
        plt.errorbar(det_fluence, inj_fluence, xerr=np.array(det_fluence_std), fmt=".")
        plt.ylabel("Injected FLUENCE")
        plt.xlabel("Detected FLUENCE")
        plt.show()
        ind = np.argsort(inj_fluence)
        self.det_fluence = det_fluence[ind]
        self.det_fluence_std = det_fluence_std[ind]
        self.inj_fluence = inj_fluence[ind]
        self.poly_fluence = p
        self.detect_error_fluence = np.mean(self.det_fluence_std[-3:])

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
        for csv in fn:
            (
                dm,
                burst_time,
                boxcar_det_snr,
                inj_snr,
                MJD,
            ) = read_positive_burst_inj(csv)
            for t, d, inj_s in zip(burst_time, dm, inj_snr):
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
                    if t_snr:
                        dm_arr = si.dm
                        t_arr = si.toas
                        truth_dm = (dm_arr < dm_hi) & (dm_arr > dm_low)
                        truth_t = (t_arr < t_hi) & (t_arr > t_low)
                        total_truth = truth_dm & truth_t
                        self.detected_truth(si, total_truth)
        for si in self.sorted_inject:
            print(f"snr {np.mean(si.snr)}",'detection frac',sum(si.detected)/len(si.detected))
        # get lists to plot
        snr = []
        det_snr = []
        det_frac = []
        detected_amplitudes = []
        detected_amplitudes_mean = []
        detected_pulses = []
        for i,s in enumerate(self.sorted_inject):
            snr.append(np.mean(s.snr))
            det_snr.append(s.det_snr)
            det_frac.append(sum(s.detected) / len(s.detected))
            detected_amplitudes.append(s.det_amp[s.detected])
            if sum(s.detected)==0:
                detected_amplitudes_mean.append(0)
            else:
                detected_amplitudes_mean.append(np.mean(s.det_amp[s.detected]))
            detected_pulses.append(s.detected)


        snr = np.array(snr)
        det_snr = np.array(det_snr)
        det_frac = np.array(det_frac)
        #

        detected_amplitudes = np.array(detected_amplitudes,dtype=object)
        detected_amplitudes_mean = np.array(detected_amplitudes_mean)
        detected_pulses = np.array(detected_pulses)

        ind = np.argsort(snr)
        self.det_frac = det_frac[ind]
        self.snr = snr[ind]
        self.detected_amplitudes = detected_amplitudes[ind]
        self.detected_amplitudes_mean = detected_amplitudes_mean[ind]
        self.det_snr = det_snr[ind]
        self.detected_pulses = detected_pulses[ind]
        #only take values of det_snr above snr of 1
        self.det_snr = self.det_snr[self.snr > 1,:]
        self.detected_pulses = self.detected_pulses[self.snr > 1,:]
        #get an array of all detected snrs
        detected_snr = self.det_snr[self.detected_pulses]
        all_det_snr = self.det_snr.flatten()


        # set the number of bins and the number of data points per bin
        num_bins = 30
        self.bin_detections(all_det_snr,detected_snr,num_bins)


        predict_x_array = np.linspace(0,np.max(self.detected_bin_midpoints),10000)
        # self.interpolate(predict_x_array,self.det_frac,self.inj_amp)
        self.poly_det_fit = self.fit_poly(x=self.detected_bin_midpoints,p=self.detected_det_frac,deg=7)
        self.predict_poly(predict_x_array,x=self.detected_bin_midpoints,p=self.detected_det_frac,plot=True,title=title)
        if hasattr(self, "base_fn"):
            plt.savefig(self.base_fn + "_fit_snr.png")
        else:
            plt.show()
        plt.close()
        inj_fit_x = np.linspace(0,np.max(self.detected_bin_midpoints),10000)
        inj_snr_fit = self.fit_poly(x=self.snr,p=self.det_frac,deg=7)
        self.predict_poly(inj_fit_x,x=self.snr,p=self.det_frac,poly=inj_snr_fit,plot=True,title=title)
        if hasattr(self, "base_fn"):
            plt.savefig(self.base_fn + "_inj_snr.png")
        else:
            plt.show()
        plt.close()

    def bin_detections(self,all_det_vals, detected_det_vals, num_bins=20,plot=False):
        # set the number of data points per bin
        num_points_per_bin = len(all_det_vals) // num_bins
        print("number of points in each bin ", num_points_per_bin)

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
        self.detected_bin_midpoints = bin_midpoints
        self.detected_det_frac = detected_det_frac
        #remove the last 2 bins
        self.detected_bin_midpoints = self.detected_bin_midpoints[:-2]
        self.detected_det_frac = self.detected_det_frac[:-2]

    def predict_poly(self,predict_x,x,p,poly=-99,start_point = 2.2,plot=False,title="polynomial fit"):
        if isinstance(poly,int):
            poly = self.poly_det_fit

        predict = np.poly1d(poly)
        p_pred = predict(predict_x)
        #find the closest index of x to start_point
        i = np.argmin(np.abs(predict_x-start_point))
        while (p_pred[i]>np.min(p)) & (p_pred[i]>0):
            i -= 1
            if i==0:
                break
        set_0 = i+1
        while (p_pred[i]<np.max(p)) & (p_pred[i]<1):
            i += 1
            if i==len(p_pred):
                break
        set_1 = i
        p_pred[:set_0] = np.min(p)
        p_pred[set_1:] = np.max(p)
        if plot:
            fig,axes = plt.subplots(1,1)
            axes.scatter(x,p,label="Raw",c='r')
            axes.plot(predict_x,p_pred,label="poly_fit")
            axes.set_xlabel("Amplitude")
            axes.set_ylabel("Det Frac")
            axes.set_title(title)
            axes.set_ylim([-0.5,1.5])
            axes.legend()

        return p_pred

    def fit_poly(self,x,p,deg=4,plot=False):
        ind_1 = np.argwhere(p==max(p))
        ind_0 = np.argwhere(p==min(p))
        ind_0 = max(ind_0)[0]-1
        ind_1 = min(ind_1)[0]+1
        if ind_0<0:
            ind_0 = 0
        if ind_1>(len(x)-1):
            ind_1 = len(x)-1
        poly = np.polyfit(x[ind_0:ind_1],p[ind_0:ind_1],deg=deg)
        return poly

    def fit_gen_log(self,p,x):
        x_pad_upper = np.linspace(0,10,1000)+np.max(x)
        x_pad_lower = np.linspace(0,-10,1000)+np.min(x)
        p_pad_upper = np.zeros(len(x_pad_upper))+1
        p_pad_lower = np.zeros(len(x_pad_lower))

        p_train = np.append(p,p_pad_lower)
        p_train = np.append(p_train,p_pad_upper)
        x_train = np.append(x,x_pad_lower)
        x_train = np.append(x_train,x_pad_upper)
        popt, pcov = opt.curve_fit(gen_log, x_train, p_train, [1, 1, 1, 1, 1,1], maxfev=int(1e6))
        x_test = np.linspace(-10,10,10000)
        y_mean = gen_log(x_test,popt[0],popt[1],popt[2],popt[3],popt[4],popt[5])
        fig,axes = plt.subplots(1,1)
        axes.scatter(x_train,p_train,label="Raw",c='r')
        axes.plot(x_test,y_mean,label="genlog_fit")
        axes.set_xlabel("Amplitude")
        axes.set_ylabel("Det Frac")
        axes.set_title("Gaussian Process fit")
        axes.legend()
        plt.show()


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
