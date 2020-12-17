"""
for producing a spectrum image from the ChromaStarPy simulation software
"""

import numpy as np
import matplotlib.pyplot as plt
# import sys
import galsim

# from astropy.io import fits
# from astropy.modeling import models, fitting
from scipy.ndimage import gaussian_filter
from scipy import constants
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter1d

from toolkit import spectrum_slicer

filename = '../ast_521/ChromaStarPy/Outputs/BF_sim-5777.0-4.44-0.0-588.0-750.0-Run.sed.txt'

with open(filename, 'r') as file:
    raw_header = file.readline()
    # skip number of wavelength points by printing out
    print(file.readline())
    # skip unit definitions
    print(file.readline())

    # read all the data columns
    raw_data = file.readlines()

    # split the lines into data
    spectrum_data = []
    for x in raw_data:
        spectrum_data.append(x.split())
    # transpose it from row delineated (line format) to column delineated
    spectrum_data = list(map(list, zip(*spectrum_data)))

    # split the data into arrays, converting to floats at the same time
    nanometers = np.array([float(number) for number in spectrum_data[0]])
    log_flux = np.array([float(number) for number in spectrum_data[1]])

spectrum_flux = 10 ** log_flux

# convert from ergs/μm to watts/nm
spectrum_flux = spectrum_flux * 10 ** -7 * 10 ** -3

# convert from flux/m^2 to photons/m^2
c = constants.speed_of_light
h = constants.h

spectrum_counts = spectrum_flux * nanometers / (h * c)

# grab a slice of data
# spectrum_start = 588
# spectrum_end = 591
spectrum_start = 654
spectrum_end = 658
nm_slice, spectrum_counts_slice = spectrum_slicer(spectrum_start,
                                                  spectrum_end,
                                                  nanometers,
                                                  spectrum_counts)

# interpolate the data to even spacing
number_pixels = nm_slice.size
nm_grid = np.linspace(nm_slice[0], nm_slice[-1], num=number_pixels)

gridded_spectrum = griddata(nm_slice, spectrum_counts_slice, xi=nm_grid, method='linear')

# filter the spectrum slice
resolution = .1 # the resolution of the spectrum in nanometers. This corresponds to FWHM of spectrum lines
fwhm = 1/np.mean(np.diff(nm_grid)) * resolution  # the fwhm in terms of data spacing
sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
filtered_counts = gaussian_filter(gridded_spectrum.data, sigma)


plt.figure('Slice of solar spectrum', figsize=(8, 6))
# plt.scatter(nm_slice, spectrum_counts_slice, s=1)
plt.plot(nm_grid, gridded_spectrum, c='tab:blue')
plt.scatter(nm_grid, filtered_counts, s=1, c='tab:orange')
plt.title('Slice of Solar Spectrum about H-α')
plt.xlabel('Nanometers')
plt.legend(['Gridded ChromaStarPy output', 'Spectrograph output'])
# plt.xlim(587, 595)


# interpolate the data a pixel grid
sim_nm_per_pixel = .05

number_pixels = int((nm_slice[-1] - nm_slice[0]) / sim_nm_per_pixel)
pixel_grid = np.linspace(nm_slice[0], nm_slice[-1], num=number_pixels)

pixel_spectrum = griddata(nm_grid, filtered_counts, xi=pixel_grid, method='linear')

plt.figure('pixel gridding', figsize=(8, 6))
plt.plot(nm_grid, filtered_counts)
plt.scatter(pixel_grid, pixel_spectrum, s=10, c='tab:orange')
plt.title('Slice of Solar Spectrum, R=6000')
plt.xlabel('Nanometers')
plt.ylabel('Photon Intensity')
plt.legend(['Spectograph output spectrum', 'Pixel Sampling'])


# expand the array, using gaussian

# this is basically a slit 10x the width
num_spacial_pixels = int(20 * resolution/sim_nm_per_pixel)
spectrum2d = np.insert(np.zeros((num_spacial_pixels, pixel_spectrum.size)),  # generate an array of zeros
                       int(num_spacial_pixels/2),                               # location to insert data, the middle
                       pixel_spectrum,                                       # spectrum to be inserted
                       axis=0)                                             # axis the spectrum is inserted along

# arbitary flux scaling, b/c the distance to the star and exposure time is arbitrary
scaling = 2e-27  # this represents (R_star/R)**2 * instrument power * exposure time
smeared_spectrum2d = gaussian_filter1d(spectrum2d * scaling, sigma=resolution/sim_nm_per_pixel*2, axis=0)


# run it through galsim to produce a sensor image
rng = galsim.BaseDeviate(5678)

spectrum_image = galsim.Image(smeared_spectrum2d, scale=1.0)  # scale is pixel/pixel
# interpolate the image so GalSim can manipulate it
spectrum_interpolated = galsim.InterpolatedImage(spectrum_image)
spectrum_interpolated.drawImage(image=spectrum_image,
                                method='phot',
                                # center=(15, 57),
                                sensor=galsim.Sensor())
galsim_sensor_image = spectrum_image.array.copy()

plt.figure('GalSim image')
plt.imshow(galsim_sensor_image[:, 5:-5], cmap='viridis')
plt.title('Ideal Sensor')

# now do it again, but with the BF effect

spectrum_image = galsim.Image(smeared_spectrum2d, scale=1.0)  # scale is pixel/pixel
# interpolate the image so GalSim can manipulate it
spectrum_interpolated = galsim.InterpolatedImage(spectrum_image)
spectrum_interpolated.drawImage(image=spectrum_image,
                                method='phot',
                                # center=(15,57),
                                offset=(0, 0),  # this needs 4 digits
                                sensor=galsim.SiliconSensor(name='lsst_e2v_50_32',
                                                            transpose=True,
                                                            rng=rng,
                                                            diffusion_factor=1.0))
galsim_bf_image = spectrum_image.array.copy()


difference_image = galsim_bf_image[:, 5:-5] - galsim_sensor_image[:, 5:-5]

trace_ideal = np.sum(galsim_sensor_image, axis=0)
trace_bf = np.sum(galsim_bf_image, axis=0)
plt.figure('Spectrum Trace', figsize=(8, 6))
# plt.plot(smeared_spectrum2d[row], label='original data')
plt.plot(pixel_grid, trace_ideal, label='No charge diffusion')
plt.plot(pixel_grid, trace_bf, label='With charge diffusion')
plt.title('Spectrum Profile Trace')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Photons (e-)')
plt.legend()


