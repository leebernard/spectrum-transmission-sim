import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import FormatStrFormatter


def z_lambda(sigma, scale_h, p0, planet_radius, mass, T, g):
    # constants
    k = 1.38e-23  # boltzmann constant k_b in J/K
    amu_kg = 1.66e-27  # kg/amu

    # set mixing ratio to 1
    xi = 1
    # set equiv scale hight to 1
    tau_eq = 1

    # calculate beta
    beta = p0 / tau_eq * np.sqrt(2*np.pi*planet_radius)
    return scale_h * np.log(xi * sigma * 1/np.sqrt(k*mass*amu_kg*T*g) * beta)


def alpha_lambda(star_radius, planet_radius, z):
    return (planet_radius / star_radius)**2 + (2 * planet_radius * z)/(star_radius**2)


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
r_earth = 6.3710e6
r_sun = 6.957e8
g_earth = 10

# using data from Gliese 876 d, pulled from Wikipedia
rad_planet = 1.65  # earth radii
rad_star = .376  # solar radii
m_planet = 6.8  # in earth masses

# fuck it, use made up shit
# data from Kepler-10c
rad_planet = 2.35
m_planet = 7.37



# reference pressure: 1 barr
p0 = 1 * 1e5

# scale height
k = 1.38e-23  # boltzmann constant k_b in J/K
amu_kg = 1.66e-27  # kg/amu
g = g_earth * m_planet/(rad_planet**2)  # m/s^2
T = 290  # K
mass = 18  # amu
H = k*T/(mass*amu_kg * g)

# scale reference pressure up
# maybe later

r_p = r_earth * rad_planet
r_star = r_sun * rad_star
baseline_depth = (r_p/r_star)**2

transit_depth = alpha_lambda(star_radius=r_star,
                             planet_radius=r_p,
                             z=z_lambda(sigma=cross_sections,
                                        scale_h=H,
                                        p0=p0,
                                        planet_radius=r_p,
                                        mass=mass,
                                        T=T,
                                        g=g)
                             )

# generate photon noise from a signal value
signal = 1.22e8
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





