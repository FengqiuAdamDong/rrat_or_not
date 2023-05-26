import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import dill
import cupy as cp
def load_detection_fn(detection_curve):
    global inj_stats
    with open(detection_curve, "rb") as inf:
        inj_stats = dill.load(inf)
    global det_error
    det_error = inj_stats.detect_error_snr
    snr_arr = np.linspace(0, 10, 1000)
    print("det error", det_error)
    detfn = p_detect(snr_arr)
    plt.plot(snr_arr, detfn)


def p_detect(snr,interp=True):
    if interp:
        interp_res = np.interp(snr,inj_stats.detected_bin_midpoints,inj_stats.detected_det_frac)
        #remmove those values below snr=1.3
        interp_res[snr<1.3] = 0
        return interp_res
    return inj_stats.predict_poly(snr,x=inj_stats.detected_bin_midpoints,p=inj_stats.detected_det_frac)

def p_detect_cupy(snr,interp=True):
    if interp:
        interp_res = cp.interp(snr,cp.array(inj_stats.detected_bin_midpoints),cp.array(inj_stats.detected_det_frac))
        #remmove those values below snr=1.3
        interp_res[snr<1.3] = 0
        return interp_res
    return inj_stats.predict_poly(snr,x=inj_stats.detected_bin_midpoints,p=inj_stats.detected_det_frac)


def p_detect_quad(snr,interp=True):
    if interp:
        if snr>1.3:
            interp_res = np.interp(snr,inj_stats.detected_bin_midpoints,inj_stats.detected_det_frac)
        else:
            return 0
        #remmove those values below snr=1.3
        return interp_res
    return inj_stats.predict_poly(snr,x=inj_stats.detected_bin_midpoints,p=inj_stats.detected_det_frac)



def lognorm_dist_quad(x, mu, sigma):
    if x>0:
        pdf = np.exp(-((np.log(x) - mu) ** 2) / (2 * sigma**2)) / (
            x * sigma * np.sqrt(2 * np.pi)
        )
    else:
        pdf = 0
    return pdf

def lognorm_dist_cupy(x, mu, sigma):
    pdf = cp.zeros(x.shape)
    pdf[x > 0] = cp.exp(-((cp.log(x[x > 0]) - mu) ** 2) / (2 * sigma**2)) / (
        x[x > 0] * sigma * cp.sqrt(2 * cp.pi)
    )
    return pdf

def lognorm_dist(x, mu, sigma):
    pdf = np.zeros(x.shape)
    pdf[x > 0] = np.exp(-((np.log(x[x > 0]) - mu) ** 2) / (2 * sigma**2)) / (
        x[x > 0] * sigma * np.sqrt(2 * np.pi)
    )
    return pdf

def second(n, mu, std, N, sigma_snr,xlim=1,x_len=20000):
    #xlim needs to be at least as large as 5 sigma_snrs though
    wide_enough = False
    while not wide_enough:
        x_lims = [-sigma_snr*xlim,xlim]
        # x_lims = [-xlim,xlim]
        amp_arr = np.linspace(x_lims[0],x_lims[1],x_len)
        LN_dist = lognorm_dist(amp_arr,mu,std)
        gaussian_error = norm.pdf(amp_arr,0,sigma_snr)
        if (gaussian_error[-1] < 1e-6)&(LN_dist[-1] < 1e-6)&(gaussian_error[0] < 1e-6):
            wide_enough = True
        else:
            xlim = xlim+1
    print(f"xlim {xlim}")
    #convolve the two arrays
    # plt.figure()
    # plt.plot(amp_arr,LN_dist)
    # plt.plot(amp_arr,gaussian_error)
    # integral_ln = np.trapz(LN_dist,amp_arr)
    # plt.title(f"xlen {x_len} xlim {xlim} integral {integral_ln}")
    conv = np.convolve(LN_dist,gaussian_error)*np.diff(amp_arr)[0]
    conv_lims = [2*x_lims[0],2*x_lims[1]]
    conv_amp_array = np.linspace(conv_lims[0],conv_lims[1],(x_len*2)-1)
    #interpolate the values for amp
    p_det = p_detect(conv_amp_array)
    likelihood = conv*(1-p_det)
    # likelihood = np.interp(amp,conv_amp_array,likelihood_conv)
    integral = np.trapz(likelihood,conv_amp_array)
    try:
        p_second_int = np.log(integral)
    except:
        import pdb
        pdb.set_trace()
    if integral > 1:
        import pdb; pdb.set_trace()
        print("Integral error", integral)
        # p_second_int = 1
    return p_second_int * (N - n)

def gaussian_cupy(x, mu, sigma):
    return cp.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * cp.sqrt(2 * cp.pi))

def gaussian(x, mu, sigma):
    return np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))

def second_integral_cupy(X):
    mu = X[0]
    std = X[1]
    xlim = 20
    x_len = 200000
     #xlim needs to be at least as large as 5 sigma_snrs though
    wide_enough = False
    sigma_snr = det_error
    while not wide_enough:
        x_lims = [-sigma_snr*xlim,xlim]
        # x_lims = [-xlim,xlim]
        amp_arr = cp.linspace(x_lims[0],x_lims[1],x_len)
        LN_dist = lognorm_dist_cupy(amp_arr,mu,std)
        gaussian_error = gaussian_cupy(amp_arr,0,sigma_snr)
        if (gaussian_error[-1] < 1e-6)&(LN_dist[-1] < 1e-6)&(gaussian_error[0] < 1e-6):
            wide_enough = True
        else:
            xlim = xlim+10
    #convolve the two arrays
    # plt.figure()
    # plt.plot(amp_arr,LN_dist)
    # plt.plot(amp_arr,gaussian_error)
    # integral_ln = np.trapz(LN_dist,amp_arr)
    # plt.title(f"xlen {x_len} xlim {xlim} integral {integral_ln}")
    conv = cp.convolve(LN_dist,gaussian_error)*cp.diff(amp_arr)[0]
    conv_lims = [2*x_lims[0],2*x_lims[1]]
    conv_amp_array = cp.linspace(conv_lims[0],conv_lims[1],(x_len*2)-1)
    #interpolate the values for amp
    p_det = p_detect_cupy(conv_amp_array)
    likelihood = conv*(1-p_det)
    # likelihood = np.interp(amp,conv_amp_array,likelihood_conv)
    integral = cp.trapz(likelihood,conv_amp_array)
    return cp.log(integral)


def second_integral(X):
    mu = X[0]
    std = X[1]
    xlim = 1
    x_len = 50000
     #xlim needs to be at least as large as 5 sigma_snrs though
    wide_enough = False
    sigma_snr = det_error
    while not wide_enough:
        if xlim > 14:
            break
        x_lims = [-sigma_snr*xlim,xlim]
        # x_lims = [-xlim,xlim]
        amp_arr = np.linspace(x_lims[0],x_lims[1],x_len)
        LN_dist = lognorm_dist(amp_arr,mu,std)
        gaussian_error = gaussian(amp_arr,0,sigma_snr)
        if (gaussian_error[-1] < 1e-6)&(LN_dist[-1] < 1e-6)&(gaussian_error[0] < 1e-6):
            wide_enough = True
        else:
            xlim = xlim+1
    print(f"xlim {xlim}")
    #convolve the two arrays
    # plt.figure()
    # plt.plot(amp_arr,LN_dist)
    # plt.plot(amp_arr,gaussian_error)
    # integral_ln = np.trapz(LN_dist,amp_arr)
    # plt.title(f"xlen {x_len} xlim {xlim} integral {integral_ln}")
    conv = np.convolve(LN_dist,gaussian_error)*np.diff(amp_arr)[0]
    conv_lims = [2*x_lims[0],2*x_lims[1]]
    conv_amp_array = np.linspace(conv_lims[0],conv_lims[1],(x_len*2)-1)
    #interpolate the values for amp
    p_det = p_detect(conv_amp_array)
    likelihood = conv*(1-p_det)
    # likelihood = np.interp(amp,conv_amp_array,likelihood_conv)
    integral = np.trapz(likelihood,conv_amp_array)
    return np.log(integral)

def second_dblquad(n, mu, std, N, sigma_snr):
    from scipy.integrate import dblquad
    def integrand(snr,snr_t,mu,std,sigma_snr):
        LN_dist = lognorm_dist_quad(snr_t,mu,std)
        gaussian_error = norm.pdf(snr-snr_t,0,sigma_snr)
        p_det = p_detect_quad(snr)
        return LN_dist*gaussian_error*(1-p_det)
    integral = dblquad(integrand,-15,15,lambda x: -15,lambda x: 15,args=(mu,std,sigma_snr),epsabs=1/(N-n))
    try:
        p_second_int = np.log(integral)
    except:
        import pdb
        pdb.set_trace()
    if integral[0] > 1:
        import pdb; pdb.set_trace()
        print("Integral error", integral)
        # p_second_int = 1

    return p_second_int * (N - n)


if __name__ == "__main__":
    load_detection_fn("inj_stats_combine_fitted.dill")
    N = 10e6
    mu = 1
    std = 0.5
    n = 835
    sigma_snr = det_error

    second(n, mu, std, N, sigma_snr,xlim=1,x_len=50000)
    # print(second_dblquad(n, mu, std, N, sigma_snr))
