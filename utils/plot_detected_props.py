import numpy as np
import matplotlib.pyplot as plt
import sys
import dill
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
mask = np.array([det_snr > 0])
det_fluence = det_fluence[mask]
det_width = det_width[mask]
det_snr = det_snr[mask]
noise_std = noise_std[mask]
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].hist(det_fluence, bins=100)
ax[0].set_title(f"Detected Fluence, total: {len(det_fluence)}")
ax[0].set_xlabel("Fluence")
ax[0].set_ylabel("Counts")
ax[1].hist(det_width, bins=100)
ax[1].set_title(f"Detected Width, total: {len(det_width)}")
ax[1].set_xlabel("Width")
ax[1].set_ylabel("Counts")
ax[2].hist(det_snr, bins=100)
ax[2].set_title(f"Detected SNR, total: {len(det_snr)}")
ax[2].set_xlabel("SNR")
ax[2].set_ylabel("Counts")
plt.show()
