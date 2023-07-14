import dill
import matplotlib.pyplot as plt
import cupy as cp
import numpy as np

def p_detect(snr,interp=True):
    if interp:
        interp_res = np.interp(snr,inj_stats.detected_bin_midpoints,inj_stats.detected_det_frac)
        # interp_res = np.interp(snr,inj_stats.detected_snr_fit,inj_stats.detected_det_frac_fit)

        #remmove those values below snr=1.3
        # interp_res[snr<1.3] = 0
        return interp_res
    return inj_stats.predict_poly(snr,x=inj_stats.detected_bin_midpoints,p=inj_stats.detected_det_frac)

def p_detect_cupy(snr,interp=True):
    interp_res = cp.interp(snr,cp.array(inj_stats.detected_bin_midpoints),cp.array(inj_stats.detected_det_frac))
    # interp_res = cp.interp(snr,cp.array(inj_stats.detected_snr_fit),cp.array(inj_stats.detected_det_frac_fit))

    #remmove those values below snr=1.3
    # interp_res[snr<1.3] = 0
    return interp_res

def load_detection_fn(detection_curve,lookup=True,plot=True):
    global inj_stats
    with open(detection_curve, "rb") as inf:
        inj_stats = dill.load(inf)
    global det_error
    det_error = inj_stats.detect_error_snr
    snr_arr = np.linspace(0, 10, 1000)
    print("det error", det_error)
    detfn = p_detect(snr_arr)
    #get the snr cutoff by finding when detfn is larger than 0.05
    snr_cutoff = snr_arr[np.where(detfn>0.006)[0][0]]
    if plot:
        plt.figure()
        plt.plot(snr_arr, detfn)
        plt.xlabel("SNR")
        plt.ylabel("Detection Fraction")
    if snr_cutoff < 1.3:
        print("WARNING SNR CUTOFF IS LESS THAN 1.3")
        snr_cutoff = 1.3
    return snr_cutoff

def logistic(x, k, x0):
    L = 1
    snr = x
    detection_fn = np.zeros(len(snr))
    snr_limit = 1
    detection_fn[(snr > -snr_limit) & (snr < snr_limit)] = L / (
        1 + np.exp(-k * (snr[(snr > -snr_limit) & (snr < snr_limit)] - x0))
    )
    detection_fn[snr >= snr_limit] = 1
    detection_fn[snr <= -snr_limit] = 0
    return detection_fn


def n_detect(snr_emit):
    # snr emit is the snr that the emitted pulse has
    p = p_detect(snr_emit)
    # simulate random numbers between 0 and 1
    rands = np.random.rand(len(p))
    # probability the random number is less than p gives you an idea of what will be detected
    detected = snr_emit[rands < p]
    return detected
