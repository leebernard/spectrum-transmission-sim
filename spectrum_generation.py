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
sigma = resolution/2.35482
filtered_sun = gaussian_filter(sun.data, sigma)

plt.figure('Full available solar spectrum')
plt.scatter(angstrom, sun.data, s=1)
plt.scatter(angstrom, filtered_sun, s=1)

# find a slice of data
start_ang = 5000
end_ang = angstrom[-1]

plot_start = np.where(angstrom == start_ang)[0][0]
plot_end = np.where(angstrom == end_ang)[0][0]
sun_slice = filtered_sun[plot_start:plot_end]
angstrom_slice = angstrom[plot_start:plot_end]

plt.figure('Slice of Filtered Solar Spectrum')
plt.scatter(angstrom_slice, sun_slice, s=1)
plt.xlim(5000, 5500)
plt.xlim(5500, 6000)
plt.xlim(6000, 6500)
plt.xlim(6500, 7000)
plt.xlim(7000, angstrom[-1])

plt.xlim(5400, 5500)  # fairly isolated feature at 5456 angs
plt.xlim(5454, 5457)

