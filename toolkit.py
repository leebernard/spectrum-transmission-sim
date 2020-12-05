"""
space for commonly needed custom functions
"""

import numpy as np


def spectrum_slicer(start_angstrom, end_angstrom, angstrom_data, spectrum_data):
    start_index = (np.abs(angstrom_data - start_angstrom)).argmin()
    end_index = (np.abs(angstrom_data - end_angstrom)).argmin()
    spectrum_slice = spectrum_data[start_index:end_index]
    angstrom_slice = angstrom_data[start_index:end_index]

    return angstrom_slice, spectrum_slice





