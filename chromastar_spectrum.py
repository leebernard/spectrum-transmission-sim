"""
for producing a spectrum image from the ChromaStarPy simulation software
"""

import numpy as np
import matplotlib.pyplot as plt
# import sys
# import galsim

# from astropy.io import fits
# from astropy.modeling import models, fitting
from scipy.ndimage import gaussian_filter
from scipy import constants
from scipy.interpolate import griddata
# from scipy.ndimage import gaussian_filter1d

from toolkit import spectrum_slicer


filename = '../ast_521/ChromaStarPy/Outputs/BF_sim-5777.0-4.44-0.0-588.0-750.0-Run.sed.txt'

with open(filename, 'r') as file:
    raw_header = file.readline()
    # skip number of wavelength points by printing out
    print(file.readline())
    # skip unit definitions
    print(file.readline())

    #read all the data columns
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

spectrum_flux = 10**log_flux

# convert from ergs/μm to watts/nm
spectrum_flux = spectrum_flux * 10**-7 * 10**-3

# convert from flux/m^2 to photons/m^2
c = constants.speed_of_light
h = constants.h

spectrum_counts = spectrum_flux * nanometers / (h*c)

'''
# filter the full spectrum
resolution = 50  # the resolution of the spectrum in nanometers. This corresponds to FWHM
sigma = resolution/(2.0 * np.sqrt(2.0 * np.log(2.0)))
filtered_flux = gaussian_filter(spectrum_flux.data, sigma)
filtered_counts = gaussian_filter(spectrum_counts.data, sigma)

plt.figure('Full available solar spectrum')
plt.scatter(nanometers, spectrum_counts, s=1)
plt.scatter(nanometers, filtered_counts, s=1)
plt.title('Slice of Solar spectrum')
plt.xlabel('Angstroms')
plt.legend(['Before smoothing', 'After smoothing'])
# plt.xlim(585, 595)
'''

# grab a slice of data
spectrum_start = 588
spectrum_end = 750
nm_slice, spectrum_slice = spectrum_slicer(spectrum_start,
                                           spectrum_end,
                                           nanometers,
                                           spectrum_counts)

# interpolate the data to even spacing
sim_nm_per_pixel = .35

number_pixels = int((nm_slice[-1] - nm_slice[0])/sim_nm_per_pixel)
pixel_nanometers = np.linspace()

xi = np.linspace(nanometers[0], nanometers[-1], 500)
test = griddata(nanometers, spectrum_counts, xi=xi)



