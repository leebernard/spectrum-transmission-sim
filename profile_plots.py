import matplotlib.pyplot as plt
import numpy as np

from astropy.io import fits
from spectrum_fitter import spectrum_gaussian_fit

fig_size = (8, 12)

# files to be compared

double_flux_file = 'spectrum_sim_gaussdouble_bftrue.fits'

base_flux_file = 'spectrum_sim_gaussian_bftrue.fits'

half_flux_file = 'spectrum_sim_gausshalf_bftrue.fits'

with fits.open(double_flux_file) as hdul:
    double_flux_image = hdul[0].data.copy()

with fits.open(base_flux_file) as hdul:
    base_flux_image = hdul[0].data.copy()

with fits.open(half_flux_file) as hdul:
    half_flux_image = hdul[0].data.copy()


# generate average profiles
mean_double_profile = np.mean(double_flux_image[:, 15:30], axis=1)
mean_double_pixels = np.arange(mean_double_profile.size)

mean_base_profile = np.mean(base_flux_image[:, 15:30], axis=1)
mean_base_pixels = np.arange(mean_base_profile.size)

mean_half_profile = np.mean(half_flux_image[:, 15:30], axis=1)
mean_half_pixels = np.arange(mean_half_profile.size)

# fit the profiles
g_double, double_fitter = spectrum_gaussian_fit(mean_double_pixels, mean_double_profile, amplitude=50000., mean=14., stddev=1.)
g_base, base_fitter = spectrum_gaussian_fit(mean_base_pixels, mean_base_profile, amplitude=50000., mean=14., stddev=1.)
g_half, half_fitter = spectrum_gaussian_fit(mean_half_pixels, mean_half_profile, amplitude=50000., mean=14., stddev=1.)


print('Results:')
print('double:', g_double.parameters)
print('baseline:', g_base.parameters)
print('half:', g_half.parameters)

double_center = g_double.parameters[1]
base_center = g_base.parameters[1]
half_center = g_half.parameters[1]

max_double = mean_double_profile.max()
max_base = mean_base_profile.max()
max_half = mean_half_profile.max()

plt.figure('mean profile comparison', figsize=(8.0, 12))

plt.subplot(411)
plt.plot(mean_double_pixels, mean_double_profile)
plt.plot(mean_double_pixels, g_double(mean_double_pixels))
plt.axvline(double_center, color='m')
plt.title('Double from baseline')
plt.ylabel('flux (e-)')
plt.legend(('Profile', 'Fit', 'Center at {:.3f}' .format(double_center)))

plt.subplot(412)
plt.plot(mean_base_pixels, mean_base_profile)
plt.plot(mean_base_pixels, g_base(mean_base_pixels))
plt.axvline(base_center, color='m')
plt.title('Baseline flux level')
plt.ylabel('flux (e-)')
plt.legend(('Profile', 'Fit', 'Center at {:.3f}' .format(base_center)))

plt.subplot(413)
plt.plot(mean_half_pixels, mean_half_profile)
plt.plot(mean_half_pixels, g_half(mean_half_pixels))
plt.axvline(base_center, color='m')
plt.title('Half of baseline')
plt.ylabel('flux (e-)')
plt.legend(('Profile', 'Fit', 'Center at {:.3f}' .format(half_center)))

plt.subplot(414)
plt.plot(mean_double_pixels, mean_double_profile/np.sum(mean_double_profile))
plt.plot(mean_base_pixels, mean_base_profile/np.sum(mean_base_profile))
plt.plot(mean_half_pixels, mean_half_profile/np.sum(mean_half_profile))
plt.title('Nomalized profiles of all three')
plt.ylabel('flux (e-)')
plt.xlabel('Pixels')
plt.legend(('{:6.0f}'.format(max_double), '{:6.0f}'.format(max_base), '{:6.0f}'.format(max_half)),
           title='Max Pixel value:')




