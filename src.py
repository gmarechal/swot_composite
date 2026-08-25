import xarray as xr
import numpy as np
import numpy.matlib

import time
import math

from scipy.interpolate import interp1d, griddata
from scipy.special import gammainc
from scipy.stats.distributions import chi2
from scipy.signal import welch

import matplotlib.pyplot as pltconfidence
import matplotlib.dates as mdates
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import ScalarFormatter
from matplotlib.collections import LineCollection


import matplotlib.pyplot as plt
from tqdm import tqdm
import cmocean

g = 9.81 #  acceleration of gravity
creator_name = 'g.marechal'



def bin_average(swot_data, duacs_data, bins):
    x = np.array(swot_data)
    y = np.array(duacs_data)

    # Remove NaNs
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    # Bin definition
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # Assign each point to a bin
    ind = np.digitize(x, bins)
    
    # Compute statistics
    y_mean = np.full(len(bin_centers), np.nan)
    y_std  = np.full(len(bin_centers), np.nan)
    n      = np.zeros(len(bin_centers), dtype=int)
    
    for i in range(len(bin_centers)):
        m = ind == i + 1
        if np.any(m):
            y_mean[i] = np.mean(y[m])
            y_std[i]  = np.std(y[m])
            n[i]      = np.sum(m)
    return y_mean, y_std, n



def compute_pdf(var, bins):

    """
    Purpose: Compute the probability density function (pdf) of a given 2D or 1D field
    ---------
    Inputs:
    ---------
    var: the snapshot or the array for the pdf
    bins: the bins axis

    
    Outputs:
    ---------
    pdf: the computed pdf in [%]
    bin_centers: the bin axis associated with the pdf
    """


    var_finite = var[np.isfinite(var)]
    hist_var, _ = np.histogram(var_finite, bins=bins)
    pdf_var = hist_var / hist_var.sum() * 100


    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    return pdf_var, bin_centers
