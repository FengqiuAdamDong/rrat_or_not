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
    from presto import rfifind

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


def grab_spectra_manual(
    gf, ts, te, mask_fn, dm, mask=True, downsamp=4, subband=256, manual=False
):
    # load the filterbank file
    g = r.FilReader(gf)
    if ts < 0:
        ts = 0
    print("start and end times", ts, te)
    tsamp = float(g.header.tsamp)
    nsamps = int((te - ts) / tsamp)
    nsamps = nsamps - nsamps % downsamp
    ssamps = int(ts / tsamp)
    # sampels to burst
    nsamps_start_zoom = int(4.1 / tsamp)
    nsamps_end_zoom = int(5.9 / tsamp)
    spec = g.read_block(ssamps, nsamps)
    # load mask
    if mask:
        print("masking data")
        data, masked_chans = maskfile(mask_fn, spec, ssamps, nsamps)
    # data.subband(256,subdm=dm,padval='median')
    data = data.dedisperse(dm)

    ts_no_ds = data[:, int(nsamps_start_zoom) : int(nsamps_end_zoom)]
    ts_no_ds = np.mean(ts_no_ds[~masked_chans, :], axis=0)
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
    if manual:
        # this gives you the location of the peak
        SNR, amp, std, loc, sigma_width = fit_FLUENCE_manual(
            dat_ts,
            tsamp * downsamp,
            6e-2,
            nsamps=int(0.9 / tsamp / downsamp),
            ds_data=waterfall_dat,
            downsamp=downsamp,
        )
        if (loc>(0.95))|(loc<0.85):
            #repeat if initial loc guess is wrong
            SNR, amp, std, loc, sigma_width = fit_FLUENCE_manual(
                dat_ts,
                tsamp * downsamp,
                6e-2,
                nsamps=int(loc / tsamp / downsamp),
                ds_data=waterfall_dat,
                downsamp=downsamp,
            )
        # std gives you how wide it is, so go to 3sigma
        # nsamp is where the burst is, so use loc
        # query ts no ds _AFTER_ loc is determined!!
        FLUENCE = fit_FLUENCE(
            ts_no_ds,
            tsamp,
            3 * sigma_width,
            nsamp=int(loc / tsamp),
            ds_data=waterfall_dat,
            plot=False,
        )
    else:
        # fit using downsampled values
        SNR, amp, std, loc, sigma_width = autofit_pulse(
            dat_ts,
            tsamp * downsamp,
            6e-2,
            nsamps=int(0.9 / tsamp / downsamp),
            ds_data=waterfall_dat,
            downsamp=downsamp,
            plot=False,
        )
        # integrate using non downsampled values!!!
        #         # query ts no ds _AFTER_ loc is determined!!

        FLUENCE = fit_FLUENCE(
            ts_no_ds,
            tsamp,
            6e-2,
            nsamp=int(loc / tsamp),
            ds_data=waterfall_dat,
            plot=False,
        )

    # recalculate the amplitude given a gaussian pulse shape
    gaussian_amp = FLUENCE / sigma_width / np.sqrt(2 * np.pi)
    print("filename:", gf, "downsample:", downsamp, "FLUENCE:", FLUENCE)
    print(f"Gaussian amp: {gaussian_amp} measured amp: {amp} width: {sigma_width}")
    return FLUENCE, std, amp, gaussian_amp, sigma_width


def autofit_pulse(ts, tsamp, width, nsamps, ds_data, downsamp, plot=True):
    # calculates the FLUENCE given a timeseries
    ind_max = nsamps
    w_bin = width / tsamp
    ts_std = np.delete(ts, range(int(ind_max - w_bin), int(ind_max + w_bin)))
    x = np.linspace(0, tsamp * len(ts), len(ts))
    x_std = np.delete(x, range(int(ind_max - w_bin), int(ind_max + w_bin)))
    # ts_std = ts
    coeffs = np.polyfit(x_std, ts_std, 10)
    poly = np.poly1d(coeffs)
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
        [mamplitude, max_time, 0.01, 0],
        args=(xind, ts_sub, std),
        method="Nelder-Mead",
    )
    fitx = max_l.x
    fitx[0] = abs(fitx[0])
    fitx[1] = abs(fitx[1])
    fitx[2] = abs(fitx[2])
    # residual
    std = np.std(ts_sub)
    Amplitude = fitx[0]
    loc = fitx[1]
    sigma_width = fitx[2]

    SNR = Amplitude / std
    # once we have calculated the location
    print(f"Fitted loc:{loc} amp:{Amplitude} std:{std} width sigma:{sigma_width}")
    if plot:
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].plot(x_std, ts_std)
        axs[0, 0].set_title("burst_removed")
        axs[1, 0].plot(x, ts_sub)
        axs[1, 0].set_title("baseline subtracted")
        axs[1, 0].plot(x, gaussian(xind, fitx[0], fitx[1], fitx[2], fitx[3]))
        axs[1, 0].set_title("baseline subtracted")
        axs[1, 1].plot(x, ts)
        axs[1, 1].set_title("OG time series")
        plt.show()

    return SNR, Amplitude, std, loc, sigma_width


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


def fit_FLUENCE_manual(ts, tsamp, width, nsamps, ds_data, downsamp):
    # calculates the FLUENCE given a timeseries
    ind_max = nsamps
    w_bin = width / tsamp
    ts_std = np.delete(ts, range(int(ind_max - w_bin), int(ind_max + w_bin)))
    x = np.linspace(0, tsamp * len(ts), len(ts))
    x_std = np.delete(x, range(int(ind_max - w_bin), int(ind_max + w_bin)))
    # ts_std = ts
    coeffs = np.polyfit(x_std, ts_std, 10)
    poly = np.poly1d(coeffs)
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
        [mamplitude, max_time, 0.01, 0],
        args=(xind, ts_sub, std),
        method="Nelder-Mead",
    )
    fitx = max_l.x
    # double the resolution
    xind_fit = np.linspace(min(xind), max(xind), len(xind) * 2)
    y_fit = gaussian(xind_fit, fitx[0], fitx[1], fitx[2], fitx[3])
    fig = plt.figure(figsize=(50, 50))
    # fig=plt.figure(figsize=(5,5))
    ax1 = plt.subplot(1, 2, 1)
    cmap = plt.get_cmap("magma")
    plt.imshow(ds_data, aspect="auto", cmap=cmap)
    ax = plt.subplot(1, 2, 2)

    k = plt.plot(xind, ts_sub)
    (my_plot,) = plt.plot(xind_fit, y_fit, lw=5)
    # ax.margins(x=0)
    axcolor = "lightgoldenrodyellow"
    pl_ax = plt.axes([0.1, 0.05, 0.78, 0.03], facecolor=axcolor)
    p_ax = plt.axes([0.1, 0.1, 0.78, 0.03], facecolor=axcolor)
    w_ax = plt.axes([0.1, 0.15, 0.78, 0.03], facecolor=axcolor)
    pl = Slider(pl_ax, "peak loc", 0.0, 3, valinit=fitx[1], valstep=1e-3)
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
        return -1, -1, -1, -1, -1

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
    # residual
    ts_sub = ts_sub - gaussian(xind, fitx[0], fitx[1], fitx[2], fitx[3])
    # plt.figure()
    # plt.plot(ts_sub)
    # plt.show()
    std = np.std(ts_sub)
    Amplitude = fitx[0]
    loc = fitx[1]
    sigma_width = fitx[2]

    SNR = Amplitude / std
    # once we have calculated the location
    print(f"Fitted loc:{loc} amp:{Amplitude} std:{std} width sigma:{sigma_width}")
    return SNR, Amplitude, std, loc, sigma_width


def logistic(x, k, x0):
    L = 1
    return L / (1 + np.exp(-k * (x - x0)))

def gen_log(x,A,B,C,M,K,v):
    return A+((K-A)/((C+np.exp(-B*(x-M)))**(1/v)))

class inject_obj:
    def __init__(
        self, fluence=1, width=1, toas=1, dm=1, downsamp=8, filfile="", mask=""
    ):
        self.fluence = fluence
        self.width = width
        self.toas = toas
        self.dm = dm
        self.filfile = filfile
        self.mask = mask
        self.det_fluence = []
        self.det_amp = []
        self.det_std = []
        self.fluence_amp = []
        self.downsamp = downsamp

    def calculate_inj_amplitude(
        self,
    ):
        self.inj_amp = self.fluence / self.width / np.sqrt(2 * np.pi)

    def repopulate(self, **kwargs):
        self.__dict__.update(kwargs)
        if ~hasattr(self, "detected"):
            self.detected = np.full(len(self.toas), False)

    def calculate_fluence_single(self, mask=True):
        ts = self.toas - 5
        te = self.toas + 5
        fluence, std, amp, gaussian_amp, sigma_width = grab_spectra_manual(
            gf=self.filfile,
            ts=ts,
            te=te,
            mask_fn=self.mask,
            dm=self.dm,
            subband=256,
            mask=True,
            downsamp=self.downsamp,
            manual=True,
        )
        # print(f"Calculated fluence:{fluence} A:{amp} S:{std} Nominal FLUENCE:{self.fluence}")
        self.det_fluence = fluence
        self.det_amp = amp
        self.fluence_amp = gaussian_amp
        self.det_std = sigma_width
        print(f"widths {self.det_std}")
        # print(fluence,amp,std,self.filfile)

    def calculate_fluence(self):

        self.calculate_inj_amplitude()
        for t, dm, inj_amp in zip(self.toas, self.dm, self.inj_amp):
            ts = t - 5
            te = t + 5
            fluence, std, amp, gaussian_amp, sigma_width = grab_spectra_manual(
                gf=self.filfile,
                ts=ts,
                te=te,
                mask_fn=self.mask,
                dm=dm,
                subband=256,
                mask=True,
                downsamp=self.downsamp,
            )
            print(
                f"injected amplitude {inj_amp} fitted amplitude {amp} fluence amplitude {gaussian_amp}"
            )
            # print(f"Calculated fluence:{fluence} A:{amp} S:{std} Nominal FLUENCE:{self.fluence}")
            self.det_fluence.append(fluence)
            self.det_amp.append(amp)
            self.det_std.append(sigma_width)
            self.fluence_amp.append(gaussian_amp)

        self.det_fluence = np.array(self.det_fluence)
        self.det_amp = np.array(self.det_amp)
        self.det_std = np.array(self.det_std)
        self.fluence_amp = np.array(self.fluence_amp)

    def return_detected(self):
        # returns the detected fluences
        return self.det_fluence[self.detected]
        # pass


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

    def get_mask_fn(self):
        # get the filenames of all the masks
        self.mask_fn = [get_mask_fn(f) for f in self.filfiles]

    def load_inj_samp(self):
        inj_data = np.load(self.inj_samp)["grid"]
        # first column time stamp, second is fluence, third column is dm last column is injected width
        self.toa_arr = inj_data[:, 0]
        self.fluence_arr = inj_data[:, 1]
        self.dm_arr = inj_data[:, 2]
        self.width_arr = inj_data[:, 3]

    def match_inj(
        self,
    ):
        # match fluence,toa,dm with certain fil file
        self.sorted_inject = []
        for f, m in zip(self.filfiles, self.mask_fn):
            sp_ = f.split("_")
            fluence_str = sp_[-1]
            fluence_str = fluence_str.strip(".fil").strip("fluence")
            fluence = np.round(float(fluence_str), 4)
            fluence_ind = np.round(self.fluence_arr, 4) == fluence
            cur_fluence = self.fluence_arr[fluence_ind]
            cur_toa = self.toa_arr[fluence_ind]
            cur_dm = self.dm_arr[fluence_ind]
            cur_width = self.width_arr[fluence_ind]
            self.sorted_inject.append(
                inject_obj(
                    fluence=cur_fluence,
                    width=cur_width,
                    toas=cur_toa,
                    dm=cur_dm,
                    downsamp=self.downsamp,
                    filfile=f,
                    mask=m,
                )
            )
        self.sorted_inject = np.array(self.sorted_inject)

    def calculate_fluence(self, multiprocessing=False):
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
            for s in self.sorted_inject:
                s.calculate_fluence()

    def amplitude_statistics(self):
        det_amp = []
        det_fluence_amp = []
        det_fluence_amp_std = []
        det_amp_std = []
        inj_amp = []
        for s in self.sorted_inject:
            s.calculate_inj_amplitude()
            fluence_amp_a = s.fluence_amp
            det_amp_a = s.det_amp
            # get rid of outliers
            f_amp_percentile = np.percentile(fluence_amp_a, [5, 95])
            d_amp_percentile = np.percentile(det_amp_a, [5, 95])
            fluence_amp = fluence_amp_a[fluence_amp_a < f_amp_percentile[1]]
            fluence_amp = fluence_amp_a[fluence_amp_a > f_amp_percentile[0]]
            det_amp_a = det_amp_a[det_amp_a < d_amp_percentile[1]]
            det_amp_a = det_amp_a[det_amp_a > d_amp_percentile[0]]
            det_amp.append(np.mean(det_amp_a))
            det_fluence_amp.append(np.mean(fluence_amp_a))
            det_amp_std.append(np.std(det_amp_a))
            det_fluence_amp_std.append(np.std(fluence_amp_a))
            # calculate amplitude from the fluence
            inj_amp.append(np.mean(s.inj_amp))

        det_amp = np.array(det_amp)
        inj_amp = np.array(inj_amp)
        det_amp_std = np.array(det_amp_std)
        det_fluence_amp_std = np.array(det_fluence_amp_std)
        p_amp = np.polyfit(det_amp, inj_amp, deg=1)
        x = np.linspace(0, 0.2)
        poly_amp = np.poly1d(p_amp)
        p_famp = np.polyfit(det_amp[6:], inj_amp[6:], deg=1)
        poly_famp = np.poly1d(p_famp)

        fig, axes = plt.subplots(1, 3)
        axes[0].plot(x, poly_amp(x))
        axes[0].errorbar(det_amp, inj_amp, xerr=np.array(det_amp_std), fmt=".")
        axes[0].set_ylabel("Injected AMP")
        axes[0].set_xlabel("Detected AMP")
        axes[1].scatter(det_amp, det_fluence_amp)
        axes[1].set_xlabel("detected amplitude fit")
        axes[1].set_ylabel("detected amplitude fluence")
        axes[2].errorbar(
            det_fluence_amp,
            inj_amp,
            xerr=det_fluence_amp_std,
            fmt=".",
            label="Fluence amp",
        )
        axes[2].plot(det_fluence_amp, poly_famp(det_fluence_amp), label="Fit")
        axes[2].set_xlabel("Detected fluence amp")
        axes[2].set_ylabel("injected amp")
        axes[2].set_title("Det amp vs injected amp")
        plt.show()
        ind = np.argsort(inj_amp)
        self.det_amp = det_amp[ind]
        self.det_amp_std = det_amp_std[ind]
        self.inj_amp = inj_amp[ind]
        self.poly_amp = p_amp
        # take the average of the last 3 for the error
        self.detect_error_amp = np.mean(self.det_amp_std[-3:])

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
        fluence_tol = 1e-4
        for csv in fn:
            (
                dm,
                burst_time,
                boxcar_det_fluence,
                inj_fluence,
                MJD,
            ) = read_positive_burst_inj(csv)
            for t, d, inj_s in zip(burst_time, dm, inj_fluence):
                # here we gotta match the values
                t_low = t - time_tol
                t_hi = t + time_tol
                dm_low = d - dm_tol
                dm_hi = d + dm_tol
                # print(t_low,t_hi,dm_low,dm_hi)
                for si in self.sorted_inject:

                    t_fluence = (np.mean(si.fluence) > (inj_s - fluence_tol)) & (
                        np.mean(si.fluence) < (inj_s + fluence_tol)
                    )
                    if t_fluence:
                        dm_arr = si.dm
                        t_arr = si.toas
                        truth_dm = (dm_arr < dm_hi) & (dm_arr > dm_low)
                        truth_t = (t_arr < t_hi) & (t_arr > t_low)
                        total_truth = truth_dm & truth_t
                        self.detected_truth(si, total_truth)
        # get lists to plot
        fluence = []
        det_frac = []
        detected_amplitudes = []
        detected_amplitudes_mean = []
        all_detected_amplitudes = []
        detected_pulses = []
        for s in self.sorted_inject:
            fluence.append(np.mean(s.fluence))
            det_frac.append(sum(s.detected) / len(s.detected))
            detected_amplitudes.append(s.det_amp[s.detected])
            detected_amplitudes_mean.append(np.mean(s.det_amp[s.detected]))
            all_detected_amplitudes.append(s.det_amp)
            detected_pulses.append(s.detected)

        fluence = np.array(fluence)
        det_frac = np.array(det_frac)
        detected_amplitudes = np.array(detected_amplitudes)
        detected_amplitudes_mean = np.array(detected_amplitudes_mean)
        detected_pulses = np.array(detected_pulses)
        all_detected_amplitudes = np.array(all_detected_amplitudes)

        ind = np.argsort(fluence)
        self.det_frac = det_frac[ind]
        self.detected_amplitudes = detected_amplitudes[ind]
        self.detected_amplitudes_mean = detected_amplitudes_mean[ind]
        nan_ind = ~np.isnan(self.detected_amplitudes_mean)
        self.inj_amp_ratio = self.inj_amp/self.detected_amplitudes_mean
        self.error_correction_log_params = self.fit_logistic(self.inj_amp_ratio[nan_ind],self.detected_amplitudes_mean[nan_ind])


        all_detected_amplitudes = all_detected_amplitudes[ind]
        detected_pulses = detected_pulses[ind]
        #remove the really low inj amplitudes because they're likely bad
        all_detected_amplitudes = all_detected_amplitudes[5:,:]
        detected_pulses = detected_pulses[5:,:]


        all_detected_pulses = all_detected_amplitudes[detected_pulses]
        all_detected_amplitudes = all_detected_amplitudes.flatten()
        # set the number of bins and the number of data points per bin
        num_bins = 80
        num_points_per_bin = len(all_detected_amplitudes) // num_bins
        print("number of points in each bin ", num_points_per_bin)

        # calculate the bin edges based on the percentiles of the data
        bin_edges = np.quantile(all_detected_amplitudes, np.linspace(0, 1, num_bins+1))

        # use the digitize function to assign data points to bins
        bin_assignments_all = np.digitize(all_detected_amplitudes, bin_edges)
        bin_assignments_detected = np.digitize(all_detected_pulses, bin_edges)

        #count the number in each bin
        hist_all, _ = np.histogram(all_detected_amplitudes, bins=bin_edges)
        hist_detected, _ = np.histogram(all_detected_pulses, bins=bin_edges)
        detected_det_frac = hist_detected/hist_all
        bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2

        fig,axes = plt.subplots(1,2)
        axes[0].scatter(bin_midpoints,detected_det_frac,label="P(Detected|amplitude_detected)")
        axes[1].scatter(bin_midpoints,hist_detected,label="the detected pulses")
        axes[1].scatter(bin_midpoints,hist_all,alpha=0.5,label="all the pulses")
        axes[1].legend()
        self.detected_bin_midpoints = bin_midpoints
        self.detected_det_frac = detected_det_frac

        predict_x_array = np.linspace(0,0.2,10000)
        # self.interpolate(predict_x_array,self.det_frac,self.inj_amp)
        poly = self.fit_poly(x=bin_midpoints,p=detected_det_frac,deg=7)
        self.poly_det_fit = poly
        self.predict_poly(predict_x_array,x=bin_midpoints,p=detected_det_frac,plot=True)
        #fit the detection curve for
        poly_true = self.fit_poly(x = self.inj_amp,p=self.det_frac,deg=4)
        self.predict_poly(predict_x_array,self.inj_amp,self.det_frac,poly=poly_true,plot=True)
        plt.show()


    def return_detected(self):
        fluence = []
        det = []
        tot = []
        for s in self.sorted_inject:
            fluence.append(s.fluence)
            det.append(sum(s.detected))
            tot.append(len(s.detected))
        return fluence, det, tot









    #these functions are for fitting the detection function. not all of them are used, but could be if required
    def interpolate(self,predict_x_array,p,x):
        from scipy.interpolate import CubicSpline
        x_pad_upper = np.linspace(1e-3,100,1000)+np.max(x)
        x_pad_lower = np.linspace(-1e-3,-100,1000)+np.min(x)
        p_pad_upper = np.zeros(len(x_pad_upper))+1
        p_pad_lower = np.zeros(len(x_pad_lower))

        p_train = np.append(p,p_pad_lower)
        p_train = np.append(p_train,p_pad_upper)
        x_train = np.append(x,x_pad_lower)
        x_train = np.append(x_train,x_pad_upper)

        ind = np.argsort(x_train)
        x_train = x_train[ind]
        p_train = p_train[ind]
        cs = CubicSpline(x_train,p_train)
        fig,axes = plt.subplots(1,1)
        axes.scatter(x,p,label="Raw",c='r')
        axes.plot(predict_x_array,cs(predict_x_array),label="interp")
        axes.set_xlabel("Amplitude")
        axes.set_ylabel("Det Frac")
        axes.set_title("interp fit")
        axes.legend()
        plt.show()

    def predict_poly(self,predict_x,x,p,poly=-99,plot=False):
        if isinstance(poly,int):
            poly = self.poly_det_fit
        ind_1 = np.argwhere(p==1)
        ind_0 = np.argwhere(p==0)
        ind_0 = max(ind_0)[0]-1
        ind_1 = min(ind_1)[0]+1

        set_0 = np.argwhere(predict_x<x[ind_0+1])
        set_1 = np.argwhere(predict_x>x[ind_1-1])

        predict = np.poly1d(poly)
        p_pred = predict(predict_x)
        p_pred[set_0] = 0
        p_pred[set_1] = 1
        #make sure 1 and 0 are the limits
        p_pred[p_pred>1] = 1
        p_pred[p_pred<0] = 0

        if plot:
            fig,axes = plt.subplots(1,1)
            axes.scatter(x,p,label="Raw",c='r')
            axes.plot(predict_x,p_pred,label="poly_fit")
            axes.set_xlabel("Amplitude")
            axes.set_ylabel("Det Frac")
            axes.set_title("Polynomial fit")
            axes.set_ylim([-0.5,1.5])
            axes.legend()
        return p_pred

    def fit_poly(self,x,p,deg=4,plot=False):
        ind_1 = np.argwhere(p==1)
        ind_0 = np.argwhere(p==0)
        ind_0 = max(ind_0)[0]-1
        ind_1 = min(ind_1)[0]+1
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
    def gaussian_process_fit(self,p,x):
        #fit the detection function using a gaussian process
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel
        kernel = 1.0 * RBF(length_scale=0.1, length_scale_bounds=(1e-4, 1e3)) + WhiteKernel(
            noise_level=1, noise_level_bounds=(1e-5, 1e1)
        )
        gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.0)
        #pad the training data at the edges
        x_pad_upper = np.linspace(0,100,1000)+np.max(x)
        x_pad_lower = np.linspace(0,-100,1000)+np.min(x)
        p_pad_upper = np.zeros(len(x_pad_upper))+1
        p_pad_lower = np.zeros(len(x_pad_lower))

        p_train = np.append(p,p_pad_lower)
        p_train = np.append(p_train,p_pad_upper)
        x_train = np.append(x,x_pad_lower)
        x_train = np.append(x_train,x_pad_upper)
        x_train = x_train.reshape(-1,1)

        #train
        gpr.fit(x_train, p_train)
        #test
        x_test = np.linspace(-10,10,1000).reshape(-1,1)
        y_mean, y_std = gpr.predict(x_test, return_std=True)

        fig,axes = plt.subplots(1,1)
        axes.scatter(x_train,p_train,label="Raw",c='r')
        axes.errorbar(x_test,y_mean,yerr=y_std,label="gp_fit")
        axes.set_xlabel("Amplitude")
        axes.set_ylabel("Det Frac")
        axes.set_title("Gaussian Process fit")
        axes.legend()
        plt.show()

    def fit_logistic(self, p, fluence):
        popt, pcov = opt.curve_fit(logistic, fluence, p, [2, 0.040], maxfev=int(1e6))
        return popt


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
        inj_stats.calculate_fluence(True)
        with open("inj_stats.dill", "wb") as of:
            dill.dump(inj_stats, of)
    else:
        with open("inj_stats.dill", "rb") as inf:
            inj_stats = dill.load(inf)
        fns = args.l
        inj_stats = inject_stats(**inj_stats.__dict__)
        inj_stats.repopulate_io()
        inj_stats.calculate_fluence_statistics()
        inj_stats.amplitude_statistics()
        inj_stats.compare(fns)

        np.save("det_fun_params", inj_stats)
        with open("inj_stats_fitted.dill", "wb") as of:
            dill.dump(inj_stats, of)
