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


# take a slice of the data at the point of interest
# fairly isolated feature at 5455.6 angs, prob a Fe I line
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
plt.plot(fe_angstroms, g(fe_angstroms), color='C1')
plt.title('Fe I feature')
plt.xlabel('Angstroms')
plt.ylabel('Power')
plt.legend(['fit profile', 'Spectrum after smoothing'])

plt.show()

