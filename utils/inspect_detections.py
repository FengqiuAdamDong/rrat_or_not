import dill
import numpy as np
import matplotlib.pyplot as plt
import sys

def process_detection_results(real_det):
    with open(real_det, "rb") as inf:
        det_class = dill.load(inf)

    det_fluence = []
    det_width = []
    det_snr = []
    noise_std = []

    for pulse_obj in det_class.sorted_pulses:
        if pulse_obj.det_amp != -1:
            det_fluence.append(pulse_obj.det_fluence)
            det_width.append(pulse_obj.det_std)
            det_snr.append(pulse_obj.det_snr)
            noise_std.append(pulse_obj.noise_std)
    det_fluence = np.array(det_fluence)
    det_width = np.array(det_width)
    det_snr = np.array(det_snr)
    noise_std = np.array(noise_std)

    return det_fluence, det_width, det_snr, noise_std

det_fluence, det_width, det_snr, noise_std = process_detection_results(sys.argv[1])

fig, ax = plt.subplots(1, 3, figsize=(12, 6))
ax[0].hist(det_width*1000, bins="auto")
ax[0].set_title("Width")
ax[0].set_xlabel("Width (ms)")
ax[1].hist(det_snr, bins="auto")
ax[1].set_title("SNR")
ax[1].set_xlabel("SNR")
ax[2].hist(det_fluence, bins="auto")
ax[2].set_title("Fluence")
ax[2].set_xlabel("Fluence")
plt.tight_layout()
plt.savefig("detection_results.png")
plt.show()