"""
This is for generating a simulated spectrum

Current goal is to take a 1-D high resolution solar spectrum, and resample it to lower pixels per
angstrom using interpolation. Then the spectrum will be mapped out to a 2-D pattern, to simulate an
idealized output from a spectrograph
"""

import numpy as np

