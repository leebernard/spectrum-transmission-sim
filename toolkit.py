"""
space for commonly needed custom functions
"""

import numpy as np

from numba import jit


def consecutive_mean(data_array):
    return (data_array[:-1] + data_array[1:]) / 2


def spectrum_slicer(start_angstrom, end_angstrom, angstrom_data, spectrum_data):
    start_index = (np.abs(angstrom_data - start_angstrom)).argmin()
    end_index = (np.abs(angstrom_data - end_angstrom)).argmin()
    spectrum_slice = spectrum_data[start_index:end_index]
    angstrom_slice = angstrom_data[start_index:end_index]

    return angstrom_slice, spectrum_slice


@jit(nopython=True)
def improved_non_uniform_tophat(wlgrid, fine_wl, fine_data):

    '''
    Bins data to a provided grid, with careful edge handling.

    This function takes a wavelength, spectrum dataset, and bins it to a
    provided wavelength grid. The wlgrid and fine_wl must have matching
    units, and can have arbitrary spacing. The wlgrid provides the edges
    of the bins.

    This function handles bin edges carefully, to preserve flux.

    NOTE: This function treats the coarse grid as bin edges, while the old
    non_uniform_tophat treated the coarse grid as bin centers

    Credit: Nat Butler, ASU

    Parameters
    ----------
    wlgrid:
        The bin edges of the wavelength grid
    fine_wl:
        wavelengths of input spectrum
    fine_data:
        values corresponding to fine_wl

    Returns
    -------
    mean_wavlengths:
        The mean wavelength value of each pixel bin
    Fint:
        The binned values
    fine_data:
        The original values before binning
    '''



    # calculate the mean of values within the given wavelength bins

    # find the nearest wavelengths corresponding to the wavelength grid boundariesj
    ii = np.searchsorted(fine_wl, wlgrid)
    # ii = fine_wl.searchsorted(wlgrid)

    # make a cumlative sum of the fine wavelengths
    Fp_cumlative = np.zeros(len(fine_wl) + 1, dtype='float64')
    Fp_cumlative[1:] = np.cumsum(fine_data)
    # Fp_cumlative[1:] = fine_data.cumsum()
    sum_data = Fp_cumlative[ii[1:]] - Fp_cumlative[ii[:-1]]
    # store how many data points were summed together
    norm = 1. * (ii[1:] - ii[:-1])

    # we would be done, but the wavelength grid might not line up perfectly with the fine wavelengths
    # mop up the bin edges by shuffling some signal between bins:
    delta = (fine_wl[ii[1:]] - wlgrid[1:]) / (fine_wl[ii[1:]] - fine_wl[ii[1:] - 1])
    sum_data -= delta * fine_data[ii[1:] - 1]
    sum_data[1:] += delta[:-1] * fine_data[ii[1:-1] - 1]
    norm -= delta
    norm[1:] += delta[:-1]

    # if necessary, clean up the first bin
    if ii[0] > 0:
        delta = (fine_wl[ii[0]] - wlgrid[0]) / (fine_wl[ii[0]] - fine_wl[ii[0] - 1])
        sum_data[0] += delta * fine_data[ii[0] - 1]
        norm[0] += delta

    # finally, take the mean of each bin by dividing the sum by the norm
    binned_data = sum_data / norm

    return binned_data, fine_data


@jit
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


