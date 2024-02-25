import numpy as np
import inject_pulses_sigpyproc as inject
import argparse

if __name__ == "__main__":
    # Set the path to the data
    parser = argparse.ArgumentParser(description='Inject individual pulses into a data file')
    parser.add_argument("--snr", type=float, help="SNR of the injected pulse")
    parser.add_argument("--width", type=float, help="Width of the injected pulse")
    parser.add_argument("--dm", type=float, help="DM of the injected pulse")

    args = parser.parse_args()
    snr = args.snr
    width = args.width
    dm = args.dm

    #load the sample_injections file
    sample_injections = np.load("sample_injections.npz",allow_pickle=True)
    grid = sample_injections['grid']
    downsamp = sample_injections['downsamp'].tolist()
    stats_window = sample_injections['stats_window'].tolist()
    duration = sample_injections['duration'].tolist()
    filename = sample_injections['filename'].tolist()
    maskfn = sample_injections['maskfn'].tolist()
    #round all the values in grid to 4sf
    snr = np.round(snr,4)
    dm = np.round(dm,5)
    width = np.round(width,5)
    snr_mask = (np.round(grid[:,1],4) == snr)
    dm_mask = (np.round(grid[:,2],5) == dm)
    width_mask = (np.round(grid[:,3],5) == width)
    mask = snr_mask & dm_mask & width_mask

    inject_grid = grid[mask,:]
    p = (dm,snr,width,filename,duration,maskfn,inject_grid,stats_window,downsamp,False)
    inject.multiprocess(p)
