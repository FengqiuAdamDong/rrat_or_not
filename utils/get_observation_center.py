import numpy as np
from sigpyproc.readers import FilReader
import sys

def get_observation_center(filename):
    fil = FilReader(filename)
    header = fil.header
    tstart = header.tstart
    tsamp = header.tsamp
    nsamples = header.nsamples
    observation_center = tstart + (nsamples/2)*tsamp
    return observation_center

if __name__ == "__main__":
    filename = sys.argv[1]
    observation_center = get_observation_center(filename)
    print(observation_center)
