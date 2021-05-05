import numpy as np

from constants import k, amu_kg, g_earth, r_earth, r_sun


def open_cross_section(filename):
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

    return wave_numbers, cross_sections

