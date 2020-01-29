"""
This is for generating a simulated spectrum

Current goal is to take a 1-D high resolution solar spectrum, and resample it to lower pixels per
angstrom using interpolation. Then the spectrum will be mapped out to a 2-D pattern, to simulate an
idealized output from a spectrograph
"""

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits

# with fits.open('sun.fits') as hdul:
hdul = fits.open('sun.fits')
sun = hdul[0]

pix_per_angstrom = sun.header['CDELT1']
intial_angstrom = sun.header['CRVAL1']

# generate an x-data set
angstrom = np.arange(0, sun.data.size, step=1) * pix_per_angstrom + intial_angstrom

plt.figure('Solar Spetrum in Angstroms')
plot_start = int(sun.data.size*.4)
plot_end = int(sun.data.size*.41)
plt.scatter(angstrom[plot_start:plot_end], sun.data[plot_start:plot_end], s=1)

