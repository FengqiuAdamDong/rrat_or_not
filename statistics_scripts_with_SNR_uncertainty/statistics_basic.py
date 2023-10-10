import dill
import matplotlib.pyplot as plt
import cupy as cp
import numpy as np
def p_detect(snr,min_snr_cutoff=1.6,flux_cal=1):
    interp_res = np.interp(snr,inj_stats.detected_bin_midpoints,inj_stats.detected_det_frac)
    # interp_res = np.interp(snr,inj_stats.detected_snr_fit,inj_stats.detected_det_frac_fit)

    #remmove those values below snr=1.3
    snr_cutoff = snr[np.where(interp_res>0)[0][0]]
    if snr_cutoff < min_snr_cutoff:
        snr_cutoff = min_snr_cutoff
    inj_stats._snr = snr*flux_cal
    inj_stats._interp_res = interp_res
    #scale the cutoff by the flux cal
    snr_cutoff = snr_cutoff*flux_cal
    inj_stats._interp_res[inj_stats._snr<snr_cutoff] = 0

    return interp_res

def p_detect_cpu(snr,interp=True):
    interp_res = np.interp(snr,np.array(inj_stats._snr),np.array(inj_stats._interp_res))
    return interp_res


def p_detect_cupy(snr,interp=True):
    interp_res = cp.interp(snr,cp.array(inj_stats._snr),cp.array(inj_stats._interp_res))
    # interp_res = cp.interp(snr,cp.array(inj_stats.detected_bin_midpoints),cp.array(inj_stats.detected_det_frac))

    # interp_res = cp.interp(snr,cp.array(inj_stats.detected_snr_fit),cp.array(inj_stats.detected_det_frac_fit))

    #remmove those values below snr=1.3
    # interp_res[snr<1.3] = 0
    return interp_res

def load_detection_fn(detection_curve,lookup=True,plot=True,min_snr_cutoff=1.6,flux_cal=1):
    global inj_stats
    global det_error
    with open(detection_curve, "rb") as inf:
        inj_stats = dill.load(inf)
    det_error = inj_stats.detect_error_snr * flux_cal
    snr_arr = np.linspace(0, 10, 10000)
    print("det error", det_error)
    detfn = p_detect(snr_arr,min_snr_cutoff=min_snr_cutoff,flux_cal=flux_cal)
    #get the snr cutoff by finding when detfn is larger than 0.05
    snr_cutoff = inj_stats._snr[np.where(detfn>0)[0][0]]
    if plot:
        plt.figure()
        plt.plot(inj_stats._snr, inj_stats._interp_res, label="Interpolated Detection Fraction")
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
