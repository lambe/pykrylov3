#!/usr/bin/env python
"""
PyKrylov3: Krylov Methods in Pure Python

PyKrylov is a library of Krylov-type iterative
methods for linear systems implemented in pure Python.
"""
import os
import sys

DOCLINES = __doc__.split("\n")

# BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
if os.path.exists('MANIFEST'):
    os.remove('MANIFEST')


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage('pykrylov3')
    config.get_version(os.path.join('pykrylov3', 'version.py'))
    return config


def setup_package():

    from numpy.distutils.core import setup

    old_path = os.getcwd()
    local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    os.chdir(local_path)
    sys.path.insert(0, local_path)
    sys.path.insert(0, os.path.join(local_path, 'pykrylov3'))  # to retrieve version

    try:
        setup(
            name='pykrylov3',
            maintainer="PyKrylov3 Developers",
            maintainer_email="andrew.b.lambe@gmail.com",
            description="Krylov Methods in Pure Python",
            long_description="\n".join(DOCLINES[2:]),
            url="http://github.com/lambe/pykrylov3/tree/master",
            download_url="http://github.com/lambe/pykrylov3/tarball/0.1.1",
            license='LICENSE',
            classifiers=[
                "Development Status :: 4 - Beta",
                "Intended Audience :: Science/Research",
                "Intended Audience :: Developers",
                "License :: OSI Approved",
                "Programming Language :: Python",
                "Topic :: Software Development",
                "Topic :: Scientific/Engineering",
                "Operating System :: Microsoft :: Windows",
                "Operating System :: POSIX",
                "Operating System :: Unix",
                "Operating System :: MacOS",
            ],
            platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
            configuration=configuration)
    finally:
        del sys.path[0]
        os.chdir(old_path)

    return


if __name__ == '__main__':
    setup_package()
