import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import FormatStrFormatter
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata

from planet_sim.transit_toolbox import alpha_lambda
from planet_sim.transit_toolbox import open_cross_section
from planet_sim.transit_toolbox import scale_h
from toolkit import spectrum_slicer


'''
generate absorption profile from cross section data

possible issues with this simulation:
cross section changing with T is not accounted for
Gravity is assumed to be constant (thin shell approximation)
Everything is 1D...

Future expansions needed:
Account for temperature structure in scale height
'''

# define some housekeeping variables
wn_start = 4000
wn_end = 10000

# open these files carefully, because they are potentially over 1Gb in size
water_data_file = './line_lists/H2O_30mbar_1500K.txt'
try:
    water_data = open_cross_section(water_data_file)
    water_wno, water_cross_sections = spectrum_slicer(wn_start, wn_end, *water_data)
finally:
    del water_data

h2_data_file = './line_lists/H2H2_CIA_30mbar_1500K.txt'
try:
    h2_data = open_cross_section(h2_data_file)
    h2_wno, h2_cross_sections = spectrum_slicer(wn_start, wn_end, *h2_data)
finally:
    del h2_data

# interpolate the two different wavenumbers to the same wavenumber
wave_numbers = np.arange(wn_start, wn_end, .1)
# convert wavenumber to wavelength
cross_wavelengths = 1e7/wave_numbers

water_cross_sections = 10**np.interp(wave_numbers, water_wno, np.log10(water_cross_sections))
h2_cross_sections = 10**np.interp(wave_numbers, h2_wno, np.log10(h2_cross_sections))

# plot them to check
plt.figure('compare_cross_section')
plt.plot(cross_wavelengths, water_cross_sections)
plt.plot(cross_wavelengths, h2_cross_sections)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Cross section (cm^2/molecule)')
plt.yscale('log')
plt.legend(('H2O', 'H2'))


# using data from Gliese 876 d, pulled from Wikipedia
rad_planet = 1.65  # earth radii
rad_star = .376  # solar radii
m_planet = 6.8  # in earth masses

# data from Kepler-10c
rad_planet = 2.35
m_planet = 7.37

# fuck it, use made up shit
# assuming relationship of r = m^0.55
rad_planet = 3.5
m_planet = 10

# reference pressure: 1 barr
p0 = 1  # bars
T = 290  # K
mass = 18  # amu


# hot jupiter time!
p0 = 1
T = 1500
water_ratio = 1500. * 1e-6  # in parts per million

# need to calculate average molecular mass of atmosphere
mass_water = 18
h2he_ratio = .17
mass_h2he = (1 - h2he_ratio)*2 + h2he_ratio*4
mass = (1 - water_ratio)*mass_h2he + water_ratio*mass_water

# temporary mass cause I don't have He cross sections yet
mass = (1 - water_ratio)*2 + water_ratio*mass_water

# baseline_depth = (r_p/r_star)**2
# scale reference pressure up
# maybe later


transit_depth = alpha_lambda(sigma_trace=water_cross_sections,
                             xi=water_ratio,
                             planet_radius=rad_planet,
                             p0=p0,
                             T=T,
                             mass=mass,
                             planet_mass=m_planet,
                             star_radius=rad_star,
                             sigma_filler=h2_cross_sections
                             )

'''Turn the data into a spectrum'''
# first, flip the data to ascending order
cross_wavelengths = np.flip(cross_wavelengths)
transit_depth = np.flip(transit_depth)

# interpolate the data to even spacing
nm_grid = np.linspace(cross_wavelengths[0], cross_wavelengths[-1], num=cross_wavelengths.size)

gridded_transit = griddata(cross_wavelengths, transit_depth, xi=nm_grid, method='linear')

# filter the spectrum slice
R = 20
resolution = np.mean(nm_grid)/R  # the resolution of the spectrum in nanometers. This corresponds to FWHM of spectrum lines
fwhm = 1/np.mean(np.diff(nm_grid)) * resolution  # the fwhm in terms of data spacing
sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
filtered_transit = gaussian_filter(gridded_transit.data, sigma)

# interpolate the data a pixel grid
sim_nm_per_pixel = resolution/2  # nyquist sample the spectrum

number_pixels = int((nm_grid[-1] - nm_grid[0]) / sim_nm_per_pixel)
pixel_wavelengths = np.linspace(cross_wavelengths[0], cross_wavelengths[-1], num=number_pixels)

pixel_transit_depth = griddata(nm_grid, filtered_transit, xi=pixel_wavelengths, method='linear')
'''end turn data into spectrum'''

# generate photon noise from a signal value
signal = 1.22e6
photon_noise = 1/np.sqrt(signal)  # calculate noise as fraction of signal
noise = np.random.normal(scale=photon_noise, size=pixel_transit_depth.size)

# add noise to the transit depth
noisey_transit_depth = pixel_transit_depth + noise

# mean spectral resolution
spec_res = resolution

plt.figure('transit depth %.2f' %spec_res, figsize=(8, 8))
plt.subplot(212)
plt.plot(cross_wavelengths, np.flip(water_cross_sections))
plt.plot(cross_wavelengths, np.flip(h2_cross_sections))
plt.title('Cross section of H2O')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Cross section (cm^2/molecule)')
plt.yscale('log')

plt.subplot(211)
plt.plot(pixel_wavelengths, pixel_transit_depth)
plt.errorbar(pixel_wavelengths, noisey_transit_depth, yerr=photon_noise, fmt='o', capsize=2.0)
plt.title('Transit depth, R= %d, water= %d ppm' % (R, water_ratio/1e-6) )
plt.legend(('Ideal', 'Photon noise'))
plt.ylabel('($R_p$/$R_{star}$)$^2$')
plt.subplot(211).yaxis.set_major_formatter(FormatStrFormatter('% 1.1e'))

'project_data/1H2-16O_6250-12500_300K_20.000000.sigma'
'project_data/1H2-16O_6250-12500_300K_100.000000.sigma'


'''Fit the data'''


# define a likelyhood function
def log_likelihood(theta, x, y, yerr, fixed):
    r_p, T = theta
    mass, p0, rad_star, m_planet = fixed
    model = alpha_lambda(sigma=x,
                         planet_radius=r_p,
                         p0=p0,
                         T=T,
                         mass=mass,
                         planet_mass=m_planet,
                         star_radius=rad_star)
    sigma2 = yerr**2
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))




