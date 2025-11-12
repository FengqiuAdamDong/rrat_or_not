#!/usr/bin/env python3
import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import curve_fit
import csv
from matplotlib.widgets import Slider, Button, RadioButtons


def gaussian(x, amp, mean, sigma, a):
    return (np.abs(amp) * np.exp(-((x - mean) ** 2) / (2 * sigma**2))) + a


def log_likelihood(X, t, y, sig):
    amp = X[0]
    mean = X[1]
    sigma = X[2]
    sig_fit = sig
    # sig_fit = X[3]
    a = X[3]
    gaussian_vals = gaussian(t, amp, mean, sigma, a)
    nloglikelihood = (gaussian_vals - y) ** 2 / (2 * sig_fit**2) - np.log(
        1 / (2.506 * sig_fit)
    )
    return np.sum(nloglikelihood)


def smoother(data, downsample=4):
    return np.mean(data.reshape(-1, downsample), axis=1)


if __name__ == "__main__":
    fn_array = glob.glob("./*")
    mjd = []
    ts_start = []
    ts_end = []
    ds = 2
    bursts_characteristics = []
    for fn in fn_array:
        if "fluxcal" in fn:
            if ".npz" in fn:
                profile = np.load(fn)
                data = profile["data"]
                ds_data = 0.25 * (
                    data[:, 0::4] + data[:, 1::4] + data[:, 2::4] + data[:, 3::4]
                )
                print(fn)
                # summed contains th profile to be fitted to a gaussian
                summed = np.mean(data, axis=0)
                summed = summed - np.median(summed)
                summed = smoother(summed, ds)

                dt = profile["dt"] * ds
                sig = np.std(summed)
                tx = np.arange(0, len(summed), 1) * dt
                peak = np.max(summed)
                peak_loc = tx[peak == summed][0]
                print(peak)
                max_l = minimize(
                    log_likelihood,
                    [peak, peak_loc, 0.01, 1],
                    args=(tx, summed, sig),
                    method="Nelder-Mead",
                )
                x = max_l.x
                y_fit = gaussian(tx, x[0], x[1], x[2], x[3])
                fig = plt.figure(figsize=(10, 10))
                ax1 = plt.subplot(1, 2, 1)
                cmap = plt.get_cmap("magma")
                plt.imshow(ds_data, aspect="auto", cmap=cmap)
                ax = plt.subplot(1, 2, 2)

                k = plt.plot(tx, summed)
                (my_plot,) = plt.plot(tx, y_fit, lw=5)
                # ax.margins(x=0)
                x = list(np.abs(v) for v in x)
                axcolor = "lightgoldenrodyellow"
                pl_ax = plt.axes([0.1, 0.05, 0.78, 0.03], facecolor=axcolor)
                p_ax = plt.axes([0.1, 0.1, 0.78, 0.03], facecolor=axcolor)
                w_ax = plt.axes([0.1, 0.15, 0.78, 0.03], facecolor=axcolor)
                pl = Slider(pl_ax, "peak loc", 0.0, 0.7, valinit=x[1], valstep=1e-5)
                p = Slider(p_ax, "peak", 0.0, 2000.0, valinit=x[0], valstep=1)
                w = Slider(w_ax, "width", 0.0, 0.1, valinit=x[2], valstep=1e-5)
                plt.tight_layout()
                plt.subplots_adjust(bottom=0.25)

                x_new = False

                def update(val):
                    peak_loc = pl.val
                    peak = p.val
                    sigma = w.val
                    a = np.mean(summed)
                    # refit with new values
                    max_l = minimize(
                        log_likelihood,
                        [peak, peak_loc, sigma, a],
                        args=(tx, summed, sig),
                        method="Nelder-Mead",
                    )
                    global x_new
                    x_new = max_l.x
                    print("new fit: ", x_new)
                    new_fit = gaussian(tx, x_new[0], x_new[1], x_new[2], x_new[3])
                    my_plot.set_ydata(new_fit)
                    fig.canvas.draw_idle()

                pl.on_changed(update)
                p.on_changed(update)
                w.on_changed(update)
                plt.show()
                print(x)
                if not isinstance(x_new, bool):
                    x = x_new
                    print(x_new)
                else:
                    pass
                print(x)
                bursts_characteristics.append(
                    {
                        "fn": fn,
                        "data": summed,
                        "dt": dt,
                        "char": x,
                        "x_order": ["peak", "peak_loc", "width", "offset"],
                    }
                )
                # print(bursts_characteristics)

    np.save("burst_characteristics", bursts_characteristics)
