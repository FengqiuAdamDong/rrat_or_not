import numpy as np
import matplotlib.pyplot as plt
from statistics import statistics_ln

detection_fn = "inj_stats_combine_fitted.dill"
stats = statistics_ln(detection_fn, plot=True, snr_cutoff=2, width_cutoff=2e-3)
stats.convolve_p_detect(plot=True)

mu_ln = -1
std_ln = 0.5
w_mu_ln = -6.5
w_std_ln = 0.5

resolutions = np.linspace(100, 100000,10000)
p_det_st_wt_arr = []
for r in resolutions:
    p_det_st_wt, true_amp_array, true_width_array  = stats.second_cupy(1, mu_ln, std_ln, w_mu_ln, w_std_ln, 1, sigma_amp=sigma_snr, sigma_w=sigma_width, a=a, lower_c=lower, upper_c=upper, amp_dist='ln', w_dist='ln', resolution=res)
    p_det_st_wt_arr.append(p_det_st_wt)

plt.plot(resolutions, p_det_st_wt_arr)
plt.xlabel("Resolution")
plt.ylabel("p_det_st_wt")
plt.show()
