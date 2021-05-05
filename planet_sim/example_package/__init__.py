# example init file from Astropy
'''

from . import core, funcs, realizations
from .core import *
from .funcs import *
from .realizations import *

__all__ = core.__all__ + realizations.__all__ + funcs.__all__

'''