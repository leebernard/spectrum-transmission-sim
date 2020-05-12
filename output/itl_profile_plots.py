import matplotlib.pyplot as plt
import numpy as np
import os

from astropy.io import fits
from spectrum_fitter import spectrum_gaussian_fit

script_dir = os.path.dirname(__file__) # <-- absolute dir the script is in

title_label = 'Gaussian Spot Image'
bffalse_name = os.path.join(script_dir, r'output/spot_nobf_1.fits')
bftrue_name = os.path.join(script_dir, r'output/spot_lsst_itl_50_32_bf_1.fits')
# bf_itl_name = os.path.join(script_dir, r'output/spot_e2v_bf.fits')

with fits.open(bffalse_name) as hdul:
    galsim_sensor_image = hdul[0].data.copy()

with fits.open(bftrue_name) as hdul:
    galsim_itl_50_image = hdul[0].data.copy()


# generate residual images
itl_residuals_1 = galsim_itl_50_image - galsim_sensor_image
# e2v_50_residuals = galsim_e2v_50_image - galsim_sensor_image
# e2v_50_comparison_residuals = galsim_e2v_50_image - galsim_e2v_image


fig_size = (9, 8)

plt.figure('GalSim itl analysis', figsize=(13, 4))
plt.suptitle(title_label)

plt.subplot(131)
plt.imshow(galsim_sensor_image, cmap='viridis')
plt.title('Gaussian Spot, Ideal Sensor')
plt.colorbar()

plt.subplot(132)
plt.imshow(galsim_itl_50_image, cmap='viridis')
plt.title('Gaussian Spot, ITL sensor')
plt.colorbar()

plt.subplot(133)
plt.imshow(itl_residuals_1, cmap='viridis', vmax=-itl_residuals_1.min())
plt.title('Residuals')
plt.colorbar()


# fit the profile
ideal_y_profile = galsim_sensor_image[:, 8]
ideal_y_pixels = np.arange(ideal_y_profile.size)

e2v_y_profile = galsim_e2v_image[:, 8]
e2v_y_pixels = np.arange(e2v_y_profile.size)

e2v_50_y_profile = galsim_e2v_50_image[:, 8]
e2v_50_y_pixels = np.arange(e2v_50_y_profile.size)

g_ideal, ideal_fitter = spectrum_gaussian_fit(ideal_y_pixels, ideal_y_profile,
                                              amplitude=ideal_y_profile.max(), mean=8., stddev=1.)
ideal_center = g_ideal.parameters[1]

g_e2v, e2v_fitter = spectrum_gaussian_fit(e2v_y_pixels, e2v_y_profile,
                                          amplitude=e2v_y_profile.max(), mean=8., stddev=1.)
e2v_center = g_e2v.parameters[1]

g_e2v_50, e2v_50_fitter = spectrum_gaussian_fit(e2v_50_y_pixels, e2v_50_y_profile,
                                                amplitude=e2v_50_y_profile.max(), mean=8., stddev=1.)
e2v_50_center = g_e2v_50.parameters[1]


plt.figure('Gaussian Profile comparison 50 update', figsize=fig_size)

plt.subplot(311)
plt.plot(ideal_y_pixels, ideal_y_profile)
plt.plot(ideal_y_pixels, g_ideal(ideal_y_pixels))
plt.axvline(ideal_center, color='m')
plt.title('Ideal sensor, profile in y direction (no BF)')
plt.ylabel('flux (e-)')
plt.legend(('Profile', 'Fit', 'Center at {:.3f}'.format(ideal_center)))

plt.subplot(312)
plt.plot(e2v_y_pixels, e2v_y_profile)
plt.plot(e2v_y_pixels, g_e2v(e2v_y_pixels))
plt.axvline(e2v_center, color='m')
plt.title('E2V sensor, profile in y direction')
plt.ylabel('flux (e-)')
plt.legend(('Profile', 'Fit', 'Center at {:.3f}'.format(e2v_center)))

plt.subplot(313)
plt.plot(e2v_50_y_pixels, e2v_50_y_profile)
plt.plot(e2v_50_y_pixels, g_e2v_50(e2v_50_y_pixels))
plt.axvline(e2v_50_center, color='m')
plt.title('E2V sensor 50 update, profile in y direction')
plt.xlabel('(pixels)')
plt.ylabel('flux (e-)')
plt.legend(('Profile', 'Fits', 'Center at {:.3f}'.format(e2v_50_center)))

plt.tight_layout()





plt.show()



