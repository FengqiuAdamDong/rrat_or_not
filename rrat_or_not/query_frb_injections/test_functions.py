import numpy as np
import matplotlib.pyplot as plt
from rrat_or_not.query_frb_injections.load_injections_data import injectionsData
import dill
from scipy.optimize import basinhopping
# from rrat_or_not.injection_scripts_fluence.injection_stats import inject_stats
from scipy import interpolate as interp

def gen_log(x, B, v, K, M,  cutoff):
    y = K / ((1 + np.exp(-B * (x - M))) ** (1 / v))
    y[x < cutoff] = 0
    return y

def flipped_gen_log(x, B, v, K, M, cutoff):
    #invert the gen_log function
    y = K - gen_log(x, B, v, K, M, 0)
    y[x > cutoff] = 0
    return y

def gen_log_2d(mesh_amp, mesh_width, K_amp, K_width, cutoff_amp, cutoff_width):
    #K_amp are the amplitude parameters, and K_width are the width parameters
    y = gen_log(mesh_amp, K_amp[0], K_amp[1], K_amp[2], K_amp[3], cutoff_amp) * \
        flipped_gen_log(mesh_width, K_width[0], K_width[1], K_width[2], K_width[3], cutoff_width)
    return y

    


if __name__ == "__main__":
    amp_range = np.linspace(0, 50, 100)
    width_range = np.linspace(0, 50, 100)

    mesh_amp, mesh_width = np.meshgrid(amp_range, width_range)
    K_amp = [0.5, 1, 1, 25]
    K_width = [0.5, 1, 1, 25]
    cutoff_amp = 0
    cutoff_width = 50
    Z = gen_log_2d(mesh_amp, mesh_width, K_amp, K_width, cutoff_amp, cutoff_width)
    plt.contourf(mesh_amp, mesh_width, Z, levels=50, cmap='viridis')
    plt.colorbar(label='Detection Fraction')
    plt.xlabel('Amplitude')
    plt.ylabel('Width')
    plt.show()
