"""
This is for generating a simulated spectrum

Current goal is to take a 1-D high resolution solar spectrum, and resample it to lower pixels per
angstrom using interpolation. Then the spectrum will be mapped out to a 2-D pattern, to simulate an
idealized output from a spectrograph
"""

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
import astropy.convolution as conv
from scipy import signal
from scipy.ndimage import gaussian_filter
from scipy.ndimage.filters import _gaussian_kernel1d
# from scipy import integrate

# with fits.open('sun.fits') as hdul:
hdul = fits.open('sun.fits')
sun = hdul[0]

angstrom_per_pix = sun.header['CDELT1']
intial_angstrom = sun.header['CRVAL1']

# generate an x-data set
angstrom = np.arange(0, sun.data.size, step=1) * angstrom_per_pix + intial_angstrom

plot_start = int(sun.data.size*0.0)
plot_end = int(sun.data.size*.15)
# take a slice of data
sun_slice = sun.data[plot_start:plot_end]
angstrom_slice = angstrom[plot_start:plot_end]
# '''
# filter_factor = 10
# x = np.linspace(-5*np.pi, 5*np.pi, num=sun_slice.size)  # generate values over 10 periods
# sinc_kernel = np.sinc(x*filter_factor)
# # normalize the kernal
# sinc_kernel = sinc_kernel/np.sum(sinc_kernel)

# generate a gaussian kernel
resolution = 50  # the resolution of the spectrum in angstroms. This corresponds to FWHM
sigma = resolution/2.35482
gauss_kernel = signal.gaussian(sun_slice.size, sigma)
# normalize
gauss_kernel = gauss_kernel * 1/np.sum(gauss_kernel)

plt.figure('kernel plot')
plt.scatter(angstrom_slice, gauss_kernel, s=2)

# convovle the signal with a low pass filter
lp_sun = signal.convolve(sun_slice, gauss_kernel, mode='same')
# '''

# filter the data
resolution = 50 # /.002  # the resolution of the spectrum in angstroms. This corresponds to FWHM
sigma = resolution/2.35482
lp_sun = gaussian_filter(sun_slice, sigma)
lw = int(4.0 * float(sigma) + 0.5)
gauss_kernel = _gaussian_kernel1d(sigma, order=0, radius=lw)

plt.figure('Solar Spetrum in Angstroms')
plt.scatter(angstrom_slice, sun_slice, s=1)
# plt.figure('filtered spectrum')
plt.scatter(angstrom_slice, lp_sun, s=2)
plt.xlim(3950, 3960)



