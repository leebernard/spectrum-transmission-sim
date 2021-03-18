"""
Functions for running the planet transit simulation
"""

import numpy as np
# import matplotlib.pyplot as plt


# global constants
k = 1.38e-23  # boltzmann constant k_b in J/K
amu_kg = 1.66e-27  # kg/amu
g_earth = 10


def gravity(mass_planet, rad_planet):
    return g_earth * mass_planet / (rad_planet ** 2)


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


def z_lambda(sigma, p0, planet_radius, mass, T, g):
    '''

    Parameters
    ----------
    sigma: array
        Absorption cross section of atmosphere species as a function of wavelength
    p0: float
        Reference pressure of atmosphere; pressure at z=0
    planet_radius: float
        Minimum radius of planet
    mass:
        mass of planet
    T:
        Effective temperature of planet
    g:
        Gravitational constant of planet. Assumes thin shell of atmosphere

    Returns
    -------
    z: float, array
        The amount by which the planet's occultation disk is increased by
        opacity of the atmosphere, as a function of wavelength.
    '''
    h = scale_h(mass, T, g)
    # set mixing ratio to 1
    xi = 1
    # set equiv scale height to 1
    tau_eq = 1

    # calculate beta
    beta = p0 / tau_eq * np.sqrt(2*np.pi*planet_radius)
    return h * np.log(xi * sigma * 1/np.sqrt(k*mass*amu_kg*T*g) * beta)


def alpha_lambda(sigma, planet_radius, p0, T, mass, g, star_radius):
    '''

    Parameters
    ----------
    sigma
    planet_radius
    p0
    T
    mass
    g
    star_radius

    Returns
    -------
    The eclipse depth as a function of wavelength
    '''

    z = z_lambda(sigma, p0, planet_radius, mass, T, g)

    return (planet_radius / star_radius)**2 + (2 * planet_radius * z)/(star_radius**2)






