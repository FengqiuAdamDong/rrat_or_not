#!/usr/bin/env python3

#collect data for code in https://iopscience.iop.org/article/10.3847/1538-4357/aaab62/pdf to process nulling fraction
#
import psrchive
import sys
import numpy as np
import matplotlib.pyplot as plt
import nulling_mcmc
import copy
def calculate_on_off_intensities(data,mask,on_range,off_range):
    #data is subint x phase
    phase = np.linspace(0,1,data.shape[1])
    on_ind = (phase>on_range[0])&(phase<on_range[1])
    off_ind = (phase>off_range[0])&(phase<off_range[1])
    on_i = []
    off_i = []
    for i in range(data.shape[0]):
        mask_profile = mask[i,:]
        profile = data[i,:]
        #check if this subint is completely masked
        if np.sum(mask_profile)==0:
            print("skipping subint")
            continue
        on_intensity = np.trapz(profile[on_ind],phase[on_ind])
        off_intensity = np.trapz(profile[off_ind],phase[off_ind])
        on_i.append(on_intensity)
        off_i.append(off_intensity)
    return np.array(on_i),np.array(off_i)

def process_archive(fn,keep_subints):
    arch = psrchive.Archive_load(fn)
    #lets get the mask
    total_subint = arch.get_nsubint()
    total_chans = arch.get_nchan()
    total_bins = arch.get_nbin()
    mask_array = np.zeros((total_subint,1,total_chans,total_bins))
    for i,integration in enumerate(arch):
        for j in range(integration.get_nchan()):
            mask_array[i,0,j,:] = integration.get_weight(j)

    arch.remove_baseline()
    arch.dedisperse()


    max_phase = arch.find_max_phase()
    #rorate to set max phase to 0.8
    while np.abs(max_phase-0.8)>0.1:
        print(max_phase)
        arch.rotate(max_phase-0.8)
        max_phase = arch.find_max_phase()

    raw_data = arch.get_data()
    #mask the data
    raw_data = np.ma.array(raw_data,mask=(mask_array==0))
    profiles = np.mean(raw_data,axis=2)
    masks = np.mean(mask_array,axis=2)
    #restructure
    profiles_np = np.zeros((profiles.shape[0],profiles.shape[2]))
    mask_np = np.zeros((masks.shape[0],masks.shape[2]))
    for i in range(profiles.shape[2]):
        profiles_np[:,i] = profiles[:,0,i]
        mask_np[:,i] = masks[:,0,i]
    #go ahead and subtract the baseline for each subint by fitting a 6th order polynomial
    off_ind_array = []
    for i in range(profiles_np.shape[0]):
        profile = copy.deepcopy(profiles_np[i,:])
        phase = np.linspace(0,1,profile.shape[0])
        #remove the on pulse from the profiles before fitting
        on_phase_edges = [0.6,0.9]
        on_ind = (phase>on_phase_edges[0])&(phase<on_phase_edges[1])
        fit = np.polyfit(phase[~on_ind],profile[~on_ind],1)
        baseline = np.polyval(fit,phase)
        #plot the fit
        profiles_np[i,:] = profile - baseline
        # plt.scatter(phase,profile,c='k')
        # plt.plot(phase,baseline)
        # plt.scatter(phase,profiles_np[i,:],c='r',alpha=0.5)
        # plt.figure()
        # plt.scatter(phase[~on_ind],profile[~on_ind],c='k')
        # plt.scatter(phase,baseline,c='r',alpha=0.5)
        # plt.show()

        #now scale the observation by the off pulse std
        off_phase_edges = [0.05,0.55]
        off_ind = (phase>off_phase_edges[0])&(phase<off_phase_edges[1])
        off_ind_array.append(profiles_np[i,off_ind])
    off_ind_array = np.array(off_ind_array)
    off_std = np.std(off_ind_array)
    profiles_np = profiles_np/off_std
    #keep on the keep subints portion of the data in the middle of profiles_np
    profiles_np = profiles_np[int(profiles_np.shape[0]*(1-keep_subints)/2):int(profiles_np.shape[0]*(1+keep_subints)/2),:]
    folded_profile = np.mean(profiles_np,axis = 0)
    phase = np.linspace(0,1,profiles_np.shape[1])
    plt.imshow(profiles_np,aspect="auto")
    plt.xlabel("phase bin")
    plt.ylabel("subint")
    plt.figure()
    plt.plot(phase,folded_profile)
    plt.xlabel("phase")
    plt.ylabel("intensity")
    plt.show()
    return profiles_np,mask_np

def maximise_fn(nf,on_bins,off_bins):
    return np.abs(np.sum(on_bins-nf*off_bins))

def ritchings1976(on,off):
    #histogram the on and off pulse intensities
    # Define the bin edges
    bin_edges = np.linspace(-0.4,0,5)

    # Create histograms for both arrays
    on_hist, _ = np.histogram(on, bins=bin_edges)
    off_hist, _ = np.histogram(off, bins=bin_edges)
    nf = np.linspace(0,1,1000)
    likelihood = []
    for n in nf:
        likelihood.append(maximise_fn(n,on_hist,off_hist))

    plt.plot(nf,likelihood)
    plt.xlabel("NF")
    plt.ylabel("likelihood")

    plt.figure()
    # Plot the histograms
    plt.hist(on, bins=bin_edges, alpha=0.5, label='On')
    plt.hist(off, bins=bin_edges, alpha=0.5, label='Off')
    plt.legend()

if __name__ == "__main__":
    #lets add arguments to the script
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('filenames', metavar='N', type=str, nargs='+')
    parser.add_argument('--on_phase_edges',type=float,nargs=2,default=[0.6,1])
    parser.add_argument('--off_phase_edges',type=float,nargs=2,default=[0.1,0.5])
    parser.add_argument('--ks',type=float,help="give the value as decimal between 0 and 1 for the percentage of subints to keep",default=0.5)
    args = parser.parse_args()
    on_i = np.array([])
    off_i = np.array([])
    for fn in args.filenames:
        profiles_np,mask_np = process_archive(fn,args.ks)
        on_temp,off_temp = calculate_on_off_intensities(profiles_np,mask_np,args.on_phase_edges,args.off_phase_edges)
        on_i = np.concatenate((on_i,on_temp))
        off_i = np.concatenate((off_i,off_temp))


    rit = ritchings1976(on_i,off_i)
    on_i_thresh = 0.13
    plt.figure()
    plt.hist(on_i,bins="auto",label="on",density=True)
    plt.hist(off_i,bins="auto",alpha=0.5,label="off",density=True)
    plt.axvline(on_i_thresh,color="k",linestyle="--")
    plt.legend()

    plt.title("on and off fluence bins")
    NP=nulling_mcmc.NullingPulsar(on_i, off_i, 2)
    means_fit, means_err, stds_fit, stds_err, weights_fit, weights_err, samples, lnprobs=NP.fit_mcmc(nwalkers=40,niter=200,ninit=50,nthreads=25)
    import corner
    labels = ["mu_null","mu_emit","sigma_null","sigma_emit","NF"]
    corner.corner(samples,labels=labels)
    plt.show()
    print(f"(on<{on_i_thresh})/total_on",sum(on_i<on_i_thresh)/len(on_i))
    import pdb; pdb.set_trace()
