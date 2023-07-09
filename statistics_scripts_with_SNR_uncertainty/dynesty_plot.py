#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import dynesty
import sys
from bayes_factor_NS_LN_no_a_single import loglikelihood
from bayes_factor_NS_LN_no_a_single import pt_Uniform_N
class dynesty_plot:
    """
    Class to plot dynesty results"""

    def __init__(self, filenames):
        self.filename = filenames

    def load_filenames(self):
        """
        Load the filenames into a list of data objects
        """
        self.data = []
        for f in self.filename:
            if f.endswith('.npz'):
                self.data.append(np.load(f,allow_pickle=True)['results'])
            elif f.endswith('.h5'):
                self.data.append(dynesty.NestedSampler.restore(f))

    def plot_corner(self, labels=None):
        """
        Plot the corner plot of the samples
        """
        from dynesty import plotting as dyplot
        means = []
        stds = []
        for d in self.data:
            dyplot.cornerplot(d.results, labels=labels,show_titles=True)
            mean,cov = dynesty.utils.mean_and_cov(d.results.samples,d.results.importance_weights())
            diag_cov = np.diag(cov)
            std = np.sqrt(diag_cov)
            means.append(cov)
            stds.append(std)
        self.means = np.array(covs)
        self.stds = np.array(stds)

    def plot_accraucy(self):
        """
        Plot the accuracy of the results
        """
        for fn,centre,errors in zip(self.filename,self.means,self.stds):
            #get the mu and sigma from the filename
if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('filenames', nargs='+', help='filenames to plot')

    args = parser.parse_args()
    filenames = args.filenames
    dp = dynesty_plot(filenames)
    dp.load_filenames()
    dp.plot_corner()
    import pdb; pdb.set_trace()
