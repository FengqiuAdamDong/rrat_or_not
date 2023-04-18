#!/usr/bin/env python3
from sigpyproc.Readers import FilReader
from inject_stats import get_mask_fn
from inject_stats import maskfile
import numpy as np
def filterbank_cutouts(filename, start, end, outname):
    """
    Extract a cutout from a filterbank file.
    Parameters
    ----------
    filename : str
        The name of the filterbank file.
    start : float
        The start time
    end : float
        The end time
    outname : str
        The name of the output file.
    """
    reader = FilReader(filename)
    start_sample = int(start//reader.header.tsamp)
    end_sample = int(end//reader.header.tsamp)
    data = reader.readBlock(start, end)
    reader.writeBlock(data, outname)
