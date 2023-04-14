import argparse
from sigpyproc.readers import FilReader
# Create an argument parser for the input file and the percentage of samples and channels to cut
parser = argparse.ArgumentParser(description='Cut the center of a radio observation using sigpyproc.')
parser.add_argument('--input_file', type=str, nargs = "+" , help='The path to the input observation file.')
parser.add_argument('--samples_percent', type=float, help='The percentage of time samples to cut from the center of the observation.',default=50)
args = parser.parse_args()

for filfile in args.input_file:
    # Load the observation file
    obs = FilReader(filfile)
    # Get the number of time samples and the number of frequency channels
    n_samples = obs.header.nsamples

    # Define the number of time samples and frequency channels to cut from the center
    n_samples_cut = int(n_samples * args.samples_percent / 100)

    # Calculate the start and end indices for the time and frequency cuts
    start_sample = int((n_samples - n_samples_cut) / 2)
    end_sample = start_sample + n_samples_cut

    # Cut the center of the observation
    obs_cut = obs.read_block(start_sample, n_samples_cut)

    from presto.filterbank import FilterbankFile, create_filterbank_file
    from presto import spectra as spec
    #grab presto header
    presto_header = FilterbankFile(filfile).header
    presto_header["nbits"] = 8
    #adjust the time in presto
    presto_header["tstart"] += start_sample * obs.header.tsamp / 86400.0
    # Save the cut observation to a new file
    output_file = filfile.replace('.fil',f"_cut_{int(args.samples_percent)}.fil")
    #save the data using presto
    create_filterbank_file(output_file, spectra=obs_cut.T, nbits=presto_header["nbits"], header=presto_header)
