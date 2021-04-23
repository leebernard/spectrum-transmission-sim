"""
space for commonly needed custom functions
"""

import numpy as np


def spectrum_slicer(start_angstrom, end_angstrom, angstrom_data, spectrum_data):
    start_index = (np.abs(angstrom_data - start_angstrom)).argmin()
    end_index = (np.abs(angstrom_data - end_angstrom)).argmin()
    spectrum_slice = spectrum_data[start_index:end_index]
    angstrom_slice = angstrom_data[start_index:end_index]

    return angstrom_slice, spectrum_slice


def instrument_non_uniform_tophat(wlgrid, wno, Fp):
    szmod = wlgrid.shape[0]

    delta = np.zeros(szmod)
    Fint = np.zeros(szmod)
    delta[0:-1] = wlgrid[1:] - wlgrid[:-1]
    delta[szmod-1] = delta[szmod-2]
    #pdb.set_trace()
    for i in range(szmod-1):
        i = i+1
        loc = np.where((1E4/wno >= wlgrid[i] - 0.5*delta[i-1]) & (1E4/wno < wlgrid[i] + 0.5*delta[i]))
        Fint[i] = np.mean(Fp[loc])
        #print(wlgrid[i]-0.5*delta[i-1], wlgrid[i]+0.5*delta[i])
        #print(1E4/wno[loc][::-1])

    loc = np.where((1E4/wno > wlgrid[0]-0.5*delta[0]) & (1E4/wno < wlgrid[0]+0.5*delta[0]))
    Fint[0] = np.mean(Fp[loc])

    return Fint, Fp


