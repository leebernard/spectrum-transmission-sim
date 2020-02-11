"""
This is for generating a simulated spectrum

Current goal is to take a 1-D high resolution solar spectrum, and resample it to lower pixels per
angstrom using interpolation. Then the spectrum will be mapped out to a 2-D pattern, to simulate an
idealized output from a spectrograph
"""

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.modeling import models, fitting
from scipy.ndimage import gaussian_filter
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
resolution = 500  # /.002  # the resolution of the spectrum in angstroms. This corresponds to FWHM
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

# take a slice of the data at the point of interest
fe_angstroms, fe_data = spectrum_slicer(5454, 5457, angstrom, filtered_sun)

# Fit the data using a Gaussian with vertical offset
gauss_init = models.Gaussian1D(amplitude=-2500., mean=5456., stddev=1.) + models.Shift(offset=10000)

fit_gauss = fitting.LevMarLSQFitter()
g = fit_gauss(gauss_init, fe_angstroms, fe_data)
# fit results
print(g.parameters)
# errors on the parameters
print(np.diag(fit_gauss.fit_info['param_cov']))
# fwhw result
fwhm = g.stddev_0.value * (2.0 * np.sqrt(2.0 * np.log(2.0)))

print(f'FWHM of fit: {fwhm: .4f}')
# cov matrix
print(fit_gauss.fit_info['param_cov'])


plt.figure('Fe I feature')
plt.scatter(fe_angstroms, fe_data, s=2)
plt.plot(fe_angstroms, g(fe_angstroms))


