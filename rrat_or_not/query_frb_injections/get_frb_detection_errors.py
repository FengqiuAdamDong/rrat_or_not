#this script gets the detection errors of the CHIME/FRB data for calibrated pulses
#
#
#
import numpy as np
import argparse
from matplotlib import pyplot as plt
from rrat_or_not.query_frb_injections.utils import get_formed_beam

if __name__ == "__main__":
    
    argparser = argparse.ArgumentParser(description="Get FRB detection errors for calibrated pulses.")
    argparser.add_argument("input_file", type=str, help="Path to the input .npx file containing data array.")

    args = argparser.parse_args()
    data_data = np.load(args.input_file, allow_pickle=True)
    peak_fluxes_data = data_data['peak_flux_arr']
    fluences_data = data_data['fluence_arr']
    #off pulse rms is actually the std
    off_pulse_rms_data = data_data['off_pulse_rms_arr']
    event_time_mjd_arr_data = data_data['event_time_mjd_arr']
    width_err_arr_data = data_data['width_err_arr']
    basename_arr = data_data['basename_arr']
    x_arr = data_data['x_arr']
    ha_deg_arr = data_data['ha_deg_arr']
    #filter based on x position
    beam_x_min, beam_x_max, beam_y_min, beam_y_max = get_formed_beam()
    #this just gets the stuff in the main beam area
    in_beam = (x_arr >= beam_x_min-0.5) & (x_arr <= beam_x_max+0.5)
    peak_fluxes = peak_fluxes_data[in_beam]
    fluences = fluences_data[in_beam]
    off_pulse_rms = off_pulse_rms_data[in_beam]
    event_time_mjd_arr = event_time_mjd_arr_data[in_beam]
    width_err_arr = width_err_arr_data[in_beam]
    basename_arr = basename_arr[in_beam]
    ha_deg_arr = ha_deg_arr[in_beam]
    x_arr = x_arr[in_beam]

    flux_limit = 0
    valid_flux = peak_fluxes > flux_limit
    peak_fluxes = peak_fluxes[valid_flux]
    fluences = fluences[valid_flux]
    off_pulse_rms = off_pulse_rms[valid_flux]
    event_time_mjd_arr = event_time_mjd_arr[valid_flux]
    width_err_arr = width_err_arr[valid_flux]
    basename_arr = basename_arr[valid_flux]
    ha_deg_arr = ha_deg_arr[valid_flux]
    x_arr = x_arr[valid_flux]

    print(basename_arr)


    print(f"Number of events after x cut: {len(peak_fluxes)}")

    #bin the peak_fluxes data
    peak_flux_bins = np.logspace(np.log10(np.min(peak_fluxes)), np.log10(np.max(peak_fluxes)), 10)
    peak_flux_bin_centers = 0.5 * (peak_flux_bins[1:] + peak_flux_bins[:-1])

    peak_flux_err_in_bin = []
    data_points_in_bin = []
    for i in range(len(peak_flux_bins) - 1):
        in_bin = (peak_fluxes >= peak_flux_bins[i]) & (peak_fluxes < peak_flux_bins[i+1])
        data_points_in_bin.append(np.sum(in_bin))
        peak_flux_noise_err = np.mean(off_pulse_rms[in_bin])
        peak_flux_main_beam_err = 0.1*peak_flux_bin_centers[i]
        peak_flux_err_in_bin.append(np.sqrt(peak_flux_noise_err**2 ))

    #save the binned data to a npz file
    np.savez('frb_detection_errors_binned.npz',sdet=peak_flux_bin_centers, peak_flux_err=peak_flux_err_in_bin, data_points_in_bin=data_points_in_bin)

    plt.figure()
    #colour by data points in bin
    plt.scatter(peak_flux_bin_centers, peak_flux_err_in_bin, c=data_points_in_bin, cmap='viridis', s=10)
    cbar = plt.colorbar()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Peak Flux (Jy)')
    plt.ylabel('Peak Flux Error (Jy)')

    plt.figure()
    plt.scatter(off_pulse_rms, peak_fluxes, c=x_arr, cmap='plasma', s=10)
    cbar = plt.colorbar()
    cbar.set_label('Beam X Position')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Off Pulse RMS (Jy)')
    plt.ylabel('Peak Flux (Jy)')

    plt.figure()
    plt.scatter(peak_fluxes, x_arr, c=np.log10(off_pulse_rms), cmap='inferno', s=10)
    cbar = plt.colorbar()
    cbar.set_label('Hour Angle (deg)')
    plt.xscale('log')
    plt.xlabel('Peak Flux (Jy)')
    plt.ylabel('Beam X Position (deg)')

    plt.figure()
    plt.scatter(event_time_mjd_arr, peak_fluxes, c=x_arr, cmap='cividis', s=10)
    cbar = plt.colorbar()
    cbar.set_label('Beam X Position')
    plt.xscale('linear')
    plt.yscale('log')
    plt.xlabel('Event Time (MJD)')
    plt.ylabel('Peak Flux (Jy)')




    plt.show()



    








