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

    # data.subband(int(subband))
    # data = data.scaled(False)
    # ds_data = ds_data.scaled(False)
    # ds_data.subband(subband)
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
        fluence, amp, std, loc, sigma_width = fit_FLUENCE_manual(
            dat_ts,
            tsamp,
            6e-2,
            nsamps=int(0.9 / tsamp / downsamp),
            ds_data=waterfall_dat,
            downsamp=downsamp,
        )
        # std gives you how wide it is, so go to 3sigma
        # nsamp is where the burst is, so use loc
        FLUENCE = fit_FLUENCE(
            ts_no_ds,
            tsamp,
            3 * sigma_width,
            nsamp=int(loc / tsamp),
            ds_data=waterfall_dat,
            plot=True,
        )
    else:
        FLUENCE = fit_FLUENCE(
            dat_ts,
            tsamp,
            6e-2,
            nsamp=int(0.9 / tsamp / downsamp),
            ds_data=waterfall_dat,
        )
        # the idea is that this is for injections so you don't need std or amp
        std = 0
        amp = 0
        sigma_width = 0

    print("filename:", gf, "downsample:", downsamp, "FLUENCE:", FLUENCE)
    return FLUENCE, std, amp, sigma_width


def fit_FLUENCE(ts, tsamp, width, nsamp, ds_data, plot=False):
    # calculates the FLUENCE given a timeseries
    w_bin = int(width / tsamp)
    x = np.linspace(0, tsamp * len(ts), len(ts))
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
    w_bin = width / tsamp / downsamp
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
    max_time = nsamps * tsamp * downsamp
    # x axis of the fit
    xind = np.array(list(range(len(ts_sub)))) * tsamp * downsamp

    max_l = minimize(
        log_likelihood,
        [mamplitude, max_time, 0.01, 0],
        args=(xind, ts_sub, std),
        method="Nelder-Mead",
    )
    fitx = max_l.x
    y_fit = gaussian(xind, fitx[0], fitx[1], fitx[2], fitx[3])
    fig = plt.figure(figsize=(50, 50))
    # fig=plt.figure(figsize=(5,5))
    ax1 = plt.subplot(1, 2, 1)
    cmap = plt.get_cmap("magma")
    plt.imshow(ds_data, aspect="auto", cmap=cmap)
    ax = plt.subplot(1, 2, 2)

    k = plt.plot(xind, ts_sub)
    (my_plot,) = plt.plot(xind, y_fit, lw=5)
    # ax.margins(x=0)
    axcolor = "lightgoldenrodyellow"
    pl_ax = plt.axes([0.1, 0.05, 0.78, 0.03], facecolor=axcolor)
    p_ax = plt.axes([0.1, 0.1, 0.78, 0.03], facecolor=axcolor)
    w_ax = plt.axes([0.1, 0.15, 0.78, 0.03], facecolor=axcolor)
    pl = Slider(pl_ax, "peak loc", 0.0, 3, valinit=fitx[1], valstep=1e-3)
    p = Slider(p_ax, "peak", 0.0, 1, valinit=fitx[0], valstep=1e-5)
    w = Slider(w_ax, "width", 0.0, 3, valinit=fitx[2], valstep=1e-3)
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
        new_fit = gaussian(xind, x_new[0], x_new[1], x_new[2], x_new[3])
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
        return -1, -1, -1

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
    fluence = Amplitude / std
    loc = fitx[1]
    sigma_width = fitx[2]
    # once we have calculated the location
    print(f"Fitted loc:{loc} amp:{Amplitude} std:{std} width sigma:{sigma_width}")
    return fluence, Amplitude, std, loc, sigma_width


def logistic(x, k, x0):
    L = 1
    return L / (1 + np.exp(-k * (x - x0)))


class inject_obj:
    def __init__(self, fluence=1, toas=1, dm=1, downsamp=8, filfile="", mask=""):
        self.fluence = fluence
        self.toas = toas
        self.dm = dm
        self.filfile = filfile
        self.mask = mask
        self.det_fluence = []
        self.det_amp = []
        self.det_std = []
        self.downsamp = downsamp

    def repopulate(self, **kwargs):
        self.__dict__.update(kwargs)
        if ~hasattr(self, "detected"):
            self.detected = np.full(len(self.toas), False)

    def calculate_fluence_single(self, mask=True):
        ts = self.toas - 5
        te = self.toas + 5
        fluence, std, amp, sigma_width = grab_spectra_manual(
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
        self.det_std = sigma_width
        print(f"widths {self.det_std}")
        # print(fluence,amp,std,self.filfile)

    def calculate_fluence(self):
        for t, dm in zip(self.toas, self.dm):
            ts = t - 5
            te = t + 5
            fluence, amp, std, sigma_width = grab_spectra_manual(
                gf=self.filfile,
                ts=ts,
                te=te,
                mask_fn=self.mask,
                dm=dm,
                subband=256,
                mask=True,
                downsamp=self.downsamp,
            )
            # print(f"Calculated fluence:{fluence} A:{amp} S:{std} Nominal FLUENCE:{self.fluence}")
            self.det_fluence.append(fluence)
            self.det_amp.append(amp)
            self.det_std.append(sigma_width)

        self.det_fluence = np.array(self.det_fluence)
        self.det_amp = np.array(self.det_amp)
        self.det_std = np.array(self.det_std)

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
        # first column time stamp, second is fluence, third column is dm
        # self.downsamp = np.load(self.inj_samp)["downsamp"]
        self.downsamp = 1
        self.toa_arr = inj_data[:, 0]
        self.fluence_arr = inj_data[:, 1]
        self.dm_arr = inj_data[:, 2]

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
            cur_toa = self.toa_arr[fluence_ind]
            cur_dm = self.dm_arr[fluence_ind]
            self.sorted_inject.append(
                inject_obj(fluence, cur_toa, cur_dm, self.downsamp, f, m)
            )
        self.sorted_inject = np.array(self.sorted_inject)

    def calculate_fluence(self, multiprocessing=False):
        import copy

        if multiprocessing:

            def run_calc(s):
                s.calculate_fluence()
                print(s.det_fluence)
                return copy.deepcopy(s)

            # for faster debugging
            # self.sorted_inject = self.sorted_inject[0:10]
            with ProcessPool(nodes=64) as p:
                self.sorted_inject = p.map(run_calc, self.sorted_inject)

        else:
            for s in self.sorted_inject:
                s.calculate_fluence()

    def calculate_fluence_statistics(self):
        det_fluence = []
        det_fluence_std = []
        inj_fluence = []
        for s in self.sorted_inject:
            det_fluence.append(np.mean(s.det_fluence))
            det_fluence_std.append(np.std(s.det_fluence))
            inj_fluence.append(s.fluence)

        det_fluence = np.array(det_fluence)*1e3
        inj_fluence = np.array(inj_fluence)*1e3
        det_fluence_std = np.array(det_fluence_std)*1e3
        p = np.polyfit(det_fluence,inj_fluence,deg=1)
        x = np.linspace(0,4)
        poly = np.poly1d(p)

        plt.figure()
        plt.plot(x,poly(x))
        plt.errorbar(det_fluence, inj_fluence, xerr=np.array(det_fluence_std), fmt=".")
        plt.ylabel("Injected FLUENCE")
        plt.xlabel("Detected FLUENCE")
        plt.show()
        return det_fluence, det_fluence_std, inj_fluence, p

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
                    t_fluence = (si.fluence > (inj_s - fluence_tol)) & (
                        si.fluence < (inj_s + fluence_tol)
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
        for s in self.sorted_inject:
            fluence.append(s.fluence)
            det_frac.append(sum(s.detected) / len(s.detected))
        # sort fluence
        fluence = np.array(fluence)
        det_frac = np.array(det_frac)
        ind = np.argsort(fluence)
        fluence = fluence[ind]
        det_frac = det_frac[ind]
        # change fluence to 1e3 to make numbers easier
        self.fit_det(det_frac, fluence * 1e3, plot=plot)
        if plot == True:
            plt.scatter(fluence * 1e3, det_frac, marker="X")
            plt.title(title)
            plt.savefig(title + "_detection_curve.png")
            plt.show()
        # get errors for each

    def return_detected(self):
        fluence = []
        det = []
        tot = []
        for s in self.sorted_inject:
            fluence.append(s.fluence)
            det.append(sum(s.detected))
            tot.append(len(s.detected))
        return fluence, det, tot

    def fit_det(self, p, fluence, plot=True):
        popt, pcov = opt.curve_fit(logistic, fluence, p, [2, 2.07], maxfev=int(1e6))
        self.logistic_params = popt
        if plot:
            plt.plot(fluence, logistic(fluence, popt[0], popt[1]))
            plt.xlabel("FLUENCE")
            plt.ylabel("Detection percentage")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", action="store_false", default=True, help="Set to do inj_stats analysis"
    )
    parser.add_argument(
        "-l",
        nargs="+",
        help="list of filterbank files or positive burst csv files",
        required=True,
    )
    args = parser.parse_args()
    do_fluence_calc = args.d
    if do_fluence_calc:
        inj_samples = "sample_injections.npz"
        filfiles = args.l
        init = {"filfiles": filfiles, "inj_samp": inj_samples}
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
        (
            det_fluence,
            det_fluence_std,
            inj_fluence,
            p,
        ) = inj_stats.calculate_fluence_statistics()
        inj_stats.compare(fns)
        np.savez(
            "det_fun_params",
            popt=inj_stats.logistic_params,
            det_error=np.mean(det_fluence_std),
            poly=p,
            # det_error=0.4
        )
        with open("inj_stats_fitted.dill", "wb") as of:
            dill.dump(inj_stats, of)
