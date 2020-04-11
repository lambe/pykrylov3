"""A Gallery of Common Matrix-Vector Products"""

from pykrylov3.gallery.gallery import *

__all__ = filter(lambda s:not s.startswith('_'), dir())
