"""
This is for generating a simulated spectrum

Current goal is to take a 1-D high resolution solar spectrum, and resample it to lower pixels per
angstrom using interpolation. Then the spectrum will be mapped out to a 2-D pattern, to simulate an
idealized output from a spectrograph
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

from astropy.io import fits
from astropy.modeling import models, fitting
from scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import convolve1d
# from scipy.ndimage.filters import _gaussian_kernel1d
# from scipy import integrate


def spectrum_slicer(start_angstrom, end_angstrom, angstrom_data, spectrum_data):
    start_index = np.where(angstrom_data == start_angstrom)[0][0]
    end_index = np.where(angstrom_data == end_angstrom)[0][0]
    spectrum_slice = spectrum_data[start_index:end_index]
    angstrom_slice = angstrom_data[start_index:end_index]

    return angstrom_slice, spectrum_slice


# with fits.open('sun.fits') as hdul:
hdul = fits.open('sun.fits')
sun = hdul[0]

angstrom_per_pix = sun.header['CDELT1']
intial_angstrom = sun.header['CRVAL1']

# generate an x-data set
angstrom = np.arange(0, sun.data.size, step=1) * angstrom_per_pix + intial_angstrom

# '''
# filter_factor = 10
# x = np.linspace(-5*np.pi, 5*np.pi, num=sun_slice.size)  # generate values over 10 periods
# sinc_kernel = np.sinc(x*filter_factor)
# # normalize the kernal
# sinc_kernel = sinc_kernel/np.sum(sinc_kernel)

'''
# this is just test data, to check the kernel the filter is using
lw = int(4.0 * float(sigma) + 0.5)
gauss_kernel = _gaussian_kernel1d(sigma, order=0, radius=lw)
'''

# filter the full spectrum
resolution = 500  # 1/.002  # the resolution of the spectrum in angstroms. This corresponds to FWHM
sigma = resolution/(2.0 * np.sqrt(2.0 * np.log(2.0)))
filtered_sun = gaussian_filter(sun.data, sigma)

plt.figure('Full available solar spectrum')
plt.scatter(angstrom, sun.data, s=1)
plt.scatter(angstrom, filtered_sun, s=1)

# find a slice of data
start_ang = 5000
end_ang = angstrom[-1]

angstrom_slice, sun_slice = spectrum_slicer(start_ang, end_ang, angstrom, filtered_sun)

plt.figure('Slice of Filtered Solar Spectrum')
plt.scatter(angstrom_slice, sun_slice, s=1)
plt.xlim(5000, 5500)
plt.xlim(5500, 6000)
plt.xlim(6000, 6500)
plt.xlim(6500, 7000)
plt.xlim(7000, angstrom[-1])

plt.xlim(5400, 5500)  # fairly isolated feature at 5455.6 angs, prob a Fe I line
plt.xlim(5454, 5457)

'''bin the 1D data into pixels'''
sim_angstroms_per_pixel = .25  # resolution of the simlulated pixel grid
bin_factor = int(sim_angstroms_per_pixel/angstrom_per_pix)

excess_data_index = int(filtered_sun.size % bin_factor)
binned_angstroms = angstrom[:-excess_data_index]
binned_spectrum = filtered_sun[:-excess_data_index]

binned_angstroms = np.reshape(binned_angstroms, (int(binned_angstroms.size/bin_factor), bin_factor))
binned_spectrum = np.reshape(binned_spectrum, (int(binned_spectrum.size/bin_factor), bin_factor))

binned_angstroms = np.mean(binned_angstroms, axis=1)
binned_spectrum = np.sum(binned_spectrum, axis=1) * sim_angstroms_per_pixel

plt.bar(binned_angstroms, binned_spectrum)

'''2D spectrum'''

# take a slice of data
start_ang = 5400
end_ang = 5500
angstrom_slice, sun_slice = spectrum_slicer(start_ang, end_ang, angstrom, filtered_sun)
# expand the array
num_spacial_pixels = int(10/sim_angstroms_per_pixel)
spectrum2d = np.insert(np.zeros((num_spacial_pixels, binned_spectrum.size)),  # generate an array of zeros
                       int(num_spacial_pixels/2),                               # location to insert data, the middle
                       binned_spectrum,                                       # spectrum to be inserted
                       axis=0)                                             # axis the spectrum is inserted along

test = gaussian_filter1d(spectrum2d, sigma=3, axis=0)

sys.getsizeof(test)

x_lower = 10000
x_upper = 10100
plt.figure('unsmeared spectrum')
plt.imshow(spectrum2d, cmap='viridis')
plt.xlim(x_lower, x_upper)

plt.figure('smeared spectrum')
from matplotlib.colors import LogNorm
plt.imshow(test, norm=LogNorm(1, test.max()), cmap='viridis')
plt.xlim(x_lower, x_upper)



