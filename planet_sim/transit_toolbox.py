"""
Functions for running the planet transit simulation
"""

import numpy as np

from scipy.ndimage import gaussian_filter

# import matplotlib.pyplot as plt
from toolkit import spectrum_slicer
from toolkit import instrument_non_uniform_tophat

# global constants
k = 1.38e-23  # boltzmann constant k_b in J/K
amu_kg = 1.66e-27  # kg/amu
g_jovian = 24.79  # m/s^2
r_jovian = 7.1492e7  # meters
r_sun = 6.957e8  # meters


def open_cross_section(filename, wn_range=None):
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

    if wn_range is None:
        return wave_numbers, cross_sections
    else:
        # wn range needs to be exactly 2 values
        # explicitly pass those two values
        wn_start, wn_end = wn_range
        return spectrum_slicer(wn_start, wn_end, wave_numbers, cross_sections)


def gravity(mass_planet, rad_planet):
    return g_jovian * mass_planet / (rad_planet ** 2)


def scale_h(mass, T, g):
    '''
    Calculates the scale height

    Parameters
    ----------
    mass: float
        average mass of atmosphere species, in atomic units
    T: float
        Average temperature of atmosphere
    g: float
        Gravitational constant

    Returns
    -------
    The scale height of an atmosphere
    '''
    return k*T/(mass*amu_kg * g)


def z_lambda(sigma_trace, xi, p0, planet_radius, mass, T, planet_mass, sigma_filler=False):
    '''

    Parameters
    ----------
    sigma_trace: array
        Absorption cross section of atmosphere species as a function of wavelength
    p0: float
        Reference pressure of atmosphere; pressure at z=0
    planet_radius: float
        Minimum radius of planet
    mass: float
        mass of trace atomic species
    T: float
        Effective temperature of planet
    planet_mass: float
        mass of the planet in jovian masses

    Returns
    -------
    z: float, array
        The amount by which the planet's occultation disk is increased by
        opacity of the atmosphere, as a function of wavelength.
    '''
    # convert planet radius to meters
    r_p = r_jovian * planet_radius
    # convert from bars to pa
    pressure = p0 * 100000

    g = gravity(planet_mass, planet_radius)
    h = scale_h(mass, T, g)

    if sigma_filler is not None:
        # calculate average cross section
        sigma = (1 - xi)*sigma_filler + xi*sigma_trace
    else:
        # set volume mixing ratio to 1
        xi = 1
        sigma = sigma_trace

    # set equiv scale height to 1
    tau_eq = 1

    # calculate beta
    beta = pressure / tau_eq * np.sqrt(2*np.pi*r_p)
    return h * np.log(sigma * 1/np.sqrt(k*mass*amu_kg*T*g) * beta)


def alpha_lambda(sigma_trace, xi, planet_radius, p0, T, mass, planet_mass, star_radius, sigma_filler=False):
    '''

    Parameters
    ----------
    sigma
    planet_radius
    p0
    T
    mass
    planet_mass
    star_radius

    Returns
    -------
    The eclipse depth as a function of wavelength
    '''
    # convert to meters
    r_planet = r_jovian * planet_radius
    r_star = r_sun * star_radius

    z = z_lambda(sigma_trace, xi, p0, planet_radius, mass, T, planet_mass, sigma_filler)

    return (r_planet / r_star)**2 + (2 * r_planet * z)/(r_star**2)


def gen_measured_transit(R, fine_wl, fine_transit):
    # filter the spectrum slice with a gaussian
    # the average resolution of the spectrum in micrometers. This corresponds to FWHM of spectrum lines
    resolution = np.mean(fine_wl)/R
    fwhm = 1/np.mean(np.diff(fine_wl)) * resolution  # the fwhm in terms of data spacing
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    filtered_transit = gaussian_filter(fine_transit.data, sigma)
    
    # interpolate the data a pixel grid
    # nyquist sample the spectrum at the blue end
    sim_um_per_pixel = resolution/2
    number_pixels = int((fine_wl[-1] - fine_wl[0]) / sim_um_per_pixel)
    pixel_wavelengths = np.linspace(fine_wl[0], fine_wl[-1], num=number_pixels)
    
    pixel_transit_depth, _ = instrument_non_uniform_tophat(pixel_wavelengths, fine_wl, filtered_transit)

    return pixel_wavelengths, pixel_transit_depth


def transit_sim(x, fixed):
    # fixed global variables
    p0 = 1
    mass_h2 = 2
    mass_water = 18

    # unpack model variables
    rad_planet, T, water_ratio = x
    # unpack known priors
    fine_wavelengths, water_cross_sections, h2_cross_sections, m_planet, rad_star, R, noise = fixed




