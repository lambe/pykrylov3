"""PyKrylov: Krylov Methods Library in Python"""

__docformat__ = 'restructuredtext'

from pykrylov3.version import version as __version__

# Imports


__all__ = list(filter(lambda s: not s.startswith('_'), dir()))
__all__.append('__version__')

__doc__ += """

Miscellaneous
-------------

    __version__  :  pykrylov version string
"""
