"""Conjugate-Gradient Squared Algorithm"""

# Ideally, tfqmr should be a subpackage of cgs but how do you do that with
# distutils without having tfqmr in a subdirectory of cgs???

from pykrylov3.cgs.cgs import *

__all__ = filter(lambda s:not s.startswith('_'), dir())
