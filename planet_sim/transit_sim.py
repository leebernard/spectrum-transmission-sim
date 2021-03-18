import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import FormatStrFormatter

from planet_sim.transit_toolbox import alpha_lambda
from planet_sim.transit_toolbox import scale_h

'''
generate absorption profile from cross section data
'''
# filename = './line_lists/1H2-16O_6250-12500_300K_20.000000.sigma'
filename = './line_lists/1H2-16O_6250-12500_300K_100.000000.sigma'
with open(filename) as file:
    raw_data = file.readlines()
    wave_numbers = []
    cross_sections = []
    for x in raw_data:
        wave_string, cross_string = x.split()
        wave_numbers.append(float(wave_string))
        cross_sections.append(float(cross_string))
    wave_numbers = np.array(wave_numbers)
    cross_sections = np.array(cross_sections)

# convert wavenumber to wavelength
cross_wavelengths = 1e7/wave_numbers

# star and planet radii
r_earth = 6.3710e6  # meters
r_sun = 6.957e8  # meters
g_earth = 10  # m/s

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
p_earth = 1 * 1e5

# scale height
k = 1.38e-23  # boltzmann constant k_b in J/K
amu_kg = 1.66e-27  # kg/amu
g = g_earth * m_planet/(rad_planet**2)  # m/s^2
T = 290  # K
mass = 18  # amu
# I need to turn this into a dedicated function
H = scale_h(mass, T, g)


# baseline_depth = (r_p/r_star)**2
# scale reference pressure up
# maybe later
p0 = 1  # bars

transit_depth = alpha_lambda(sigma=cross_sections,
                             planet_radius=rad_planet,
                             p0=p0,
                             T=T,
                             mass=mass,
                             planet_mass=m_planet,
                             star_radius=rad_star)

# generate photon noise from a signal value
signal = 1.22e6
photon_noise = 1/np.sqrt(signal)  # calculate noise as fraction of signal
noise = np.random.normal(scale=photon_noise, size=transit_depth.size)

# add noise to the transit depth
noisey_transit_depth = transit_depth + noise

# mean spectral resolution
spec_res = np.mean(np.diff(np.flip(cross_wavelengths)))

plt.figure('transit depth %.2f' %spec_res, figsize=(8, 8))
plt.subplot(212)
plt.plot(cross_wavelengths, cross_sections)
plt.title('Cross section of H2O')
plt.xlabel('Wavelength (nm)')
plt.ylabel('cm^2/molecule')

plt.subplot(211)
plt.plot(cross_wavelengths, transit_depth)
plt.plot(cross_wavelengths, noisey_transit_depth)
plt.title('Transit depth, resolution %.2f nm' %spec_res)
plt.legend(('Ideal', 'Photon noise'))
plt.ylabel('($R_p$/$R_{star}$)$^2$')
plt.subplot(211).yaxis.set_major_formatter(FormatStrFormatter('% 1.1e'))

'project_data/1H2-16O_6250-12500_300K_20.000000.sigma'
'project_data/1H2-16O_6250-12500_300K_100.000000.sigma'


'''Fit the data'''


# define a likelyhood function
def log_likelihood(theta, x, y, yerr, fixed):
    r_p, p0, H, T = theta
    mass, g, rad_star = fixed
    model = alpha_lambda(sigma=x,
                         planet_radius=r_p,
                         scale_h=H,
                         p0=p0,
                         T=T,
                         mass=mass,
                         g=g,
                         star_radius=rad_star)
    sigma2 = yerr**2
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))




