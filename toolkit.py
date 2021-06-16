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


def instrument_non_uniform_tophat(wlgrid, fine_wl, Fp):
    '''
    This function takes a wavelength, spectrum dataset, and bins it to a
    provided wavelength grid. The wlgrid and fine_wl must have matching units, and can
    have arbitrary spacing.

    Parameters
    ----------
    wlgrid:
        Wave length grid to interpolate to, in microns
    fine_wl:
        wavelengths of input spectrum
    Fp:
        values corresponding to fine_wl

    Returns
    -------
    Fint:
        The binned values
    Fp:
        The original values before binning
    '''
    # pull the size of the wl grid
    szmod = wlgrid.shape[0]

    delta = np.zeros(szmod)
    Fint = np.zeros(szmod)
    # extract the difference between each wavelength place
    delta[0:-1] = wlgrid[1:] - wlgrid[:-1]
    # fill in the last place with the 2nd-to-last
    delta[szmod-1] = delta[szmod-2]
    #pdb.set_trace()
    for i in range(szmod-1):
        i = i+1

        # find wavenumber locations within .5 delta of desired wavelength
        loc = np.where((fine_wl >= wlgrid[i] - 0.5*delta[i-1]) & (fine_wl < wlgrid[i] + 0.5*delta[i]))
        # take the mean of those locations, and store it
        Fint[i] = np.mean(Fp[loc])
        #print(wlgrid[i]-0.5*delta[i-1], wlgrid[i]+0.5*delta[i])
        #print(1E4/wno[loc][::-1])

    # fill in the missing location
    loc = np.where((fine_wl > wlgrid[0]-0.5*delta[0]) & (fine_wl < wlgrid[0]+0.5*delta[0]))
    Fint[0] = np.mean(Fp[loc])

    return Fint, Fp


