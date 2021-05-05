import numpy as np

import z_lambda
from constants import k, amu_kg, g_earth, r_earth, r_sun


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
    r_planet = r_earth * planet_radius
    r_star = r_sun * star_radius

    z = z_lambda(sigma_trace, xi, p0, planet_radius, mass, T, planet_mass, sigma_filler)

    return (r_planet / r_star)**2 + (2 * r_planet * z)/(r_star**2)


