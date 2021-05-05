import numpy as np

from constants import k, amu_kg, g_earth, r_earth, r_sun


def gravity(mass_planet, rad_planet):
    return g_earth * mass_planet / (rad_planet ** 2)

