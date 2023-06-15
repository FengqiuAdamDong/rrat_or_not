import numpy as np
import matplotlib.pyplot as plt
import sys
obs_t = 1
p = 1
def N_to_pfrac(x):
    total = obs_t / p
    return (1-(x / total))

def pfrac_to_N(x):
    total = obs_t / p
    return total * x


def plot_mat_ln(
    mat, N_arr, mu_arr, std_arr, title="plot"):
    # plot corner plot
    max_likelihood_ln = np.max(mat)
    mat = np.exp(mat - np.max(mat))  # *np.exp(max_likelihood_ln)
    posterior_N = np.trapz(np.trapz(mat, mu_arr, axis=0), std_arr, axis=0)
    posterior_std = np.trapz(np.trapz(mat, mu_arr, axis=0), N_arr, axis=1)
    posterior_mu = np.trapz(np.trapz(mat, std_arr, axis=1), N_arr, axis=1)
    # posterior_N = np.max(mat, axis=(0, 1))
    # posterior_std = np.max(mat, axis=(0, 2))
    # posterior_mu = np.max(mat, axis=(1, 2))

    d_pos_mu_N = np.trapz(mat, std_arr, axis=1)
    d_pos_std_N = np.trapz(mat, mu_arr, axis=0)
    d_pos_mu_std = np.trapz(mat, N_arr, axis=-1)
    # d_pos_mu_N = np.max(mat, axis=1)
    # d_pos_std_N = np.max(mat, axis=0)
    # d_pos_mu_std = np.max(mat, axis=-1)
    fig, ax = plt.subplots(3, 3)
    fig.suptitle(title)
    ax[0, 0].plot(mu_arr, posterior_mu/np.max(posterior_mu))
    max_mu = mu_arr[np.argmax(posterior_mu)]
    # ax[0, 0].plot(posterior_mu)

    ax[1, 0].pcolormesh(mu_arr, std_arr, d_pos_mu_std.T)
    # ax[1, 0].imshow(d_pos_mu_std.T, aspect="auto")
    ax[1, 1].plot(std_arr, posterior_std/np.max(posterior_std))
    max_std = std_arr[np.argmax(posterior_std)]
    # ax[1, 1].plot(posterior_std)

    ax[2, 0].pcolormesh(mu_arr, N_arr, d_pos_mu_N.T)
    # ax[2, 0].imshow(d_pos_mu_N.T, aspect="auto")
    ax[2, 1].pcolormesh(std_arr, N_arr, d_pos_std_N.T)
    # ax[2, 1].imshow(d_pos_std_N.T, aspect="auto")
    ax[2, 2].plot(N_arr, posterior_N/np.max(posterior_N))
    max_N = N_arr[np.argmax(posterior_N)]
    ax[2, 0].set_xlabel("mu")
    ax[2, 0].set_ylabel("N")
    ax[1, 0].set_ylabel("std")
    ax[2, 1].set_xlabel("std")
    ax[2, 2].set_xlabel("N")

    N_frac1 = ax[2, 0].secondary_yaxis("right", functions=(N_to_pfrac, N_to_pfrac))
    N_frac2 = ax[2, 1].secondary_yaxis("right", functions=(N_to_pfrac, N_to_pfrac))
    N_frac2.set_ylabel("Nulling Fraction")
    N_frac3 = ax[2, 2].secondary_xaxis("top", functions=(N_to_pfrac, N_to_pfrac))
    N_frac3.set_xlabel("Nulling Fraction")

    fig.delaxes(ax[0, 1])
    fig.delaxes(ax[0, 2])
    fig.delaxes(ax[1, 2])
    plt.tight_layout()
    return max_mu, max_std, max_N

def me(mean, var):
    mu = np.log(mean**2/np.sqrt(var+mean**2))
    std = np.sqrt(np.log(var/mean**2+1))
    return mu, std


# Load data
data = np.load(sys.argv[1],allow_pickle=True)
mu_arr = data['mu_arr']
std_arr = data['std_arr']
N_arr = data['N_arr']
mat = data['mat']
plot_mat_ln(mat, N_arr, mu_arr, std_arr, title=sys.argv[1])
plt.show()
import pdb; pdb.set_trace()
