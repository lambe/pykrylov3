# Test case for CG with a diagonally dominant matrix
import sys

import numpy as np
from math import sqrt, sin, pi

from pykrylov3.gallery import Poisson1dMatvec, Poisson2dMatvec
from pykrylov3.linop import LinearOperator
from pykrylov3.cg import CG
from pykrylov3.tools import machine_epsilon


class TestPoisson1dCase:

    def setup(self):
        self.n = [10, 20, 100, 1000, 5000, 10000]
        self.eps = machine_epsilon()
        self.fmt = '%6d  %7d  %8.2e  %8.2e\n'
        hdrfmt = '%6s  %7s  %8s  %8s\n'
        hdr = hdrfmt % ('Size', 'Matvec', 'Resid', 'Error')
        print('\n  Poisson1D tests\n')
        print(hdr + '-' * len(hdr) + '\n')

    def teardown(self):
        return

    def testPoisson1D(self):
        # Solve 1D Poisson systems of various sizes
        for n in self.n:

            lmbd_min = 4.0 * sin(pi/2.0/n) ** 2
            lmbd_max = 4.0 * sin((n-1)*pi/2.0/n) ** 2
            cond = lmbd_max/lmbd_min
            tol = cond * self.eps

            A = LinearOperator(n, n,
                               lambda x: Poisson1dMatvec(x),
                               symmetric=True)
            e = np.ones(n)
            rhs = A * e
            cg = CG(A, matvec_max=2*n, outputStream=sys.stderr)
            cg.solve(rhs)
            err = np.linalg.norm(e-cg.bestSolution)/sqrt(n)
            print(self.fmt % (n, cg.nMatvec, cg.residNorm, err))
            assert np.allclose(e, cg.bestSolution, rtol=tol) == True


class TestPoisson2dCase:

    def setup(self):
        self.n = [10, 20, 100, 500]
        self.eps = machine_epsilon()
        self.fmt = '%6d  %7d  %8.2e  %8.2e\n'
        hdrfmt = '%6s  %7s  %8s  %8s\n'
        hdr = hdrfmt % ('Size', 'Matvec', 'Resid', 'Error')
        print('\n  Poisson2D tests\n')
        print(hdr + '-' * len(hdr) + '\n')

    def teardown(self):
        return

    def testPoisson2D(self):
        # Solve 1D Poisson systems of various sizes
        for n in self.n:

            h = 1.0/n
            lmbd_min = 4.0/h/h*(sin(pi*h/2.0)**2 + sin(pi*h/2.0)**2)
            lmbd_max = 4.0/h/h*(sin((n-1)*pi*h/2.0)**2 + sin((n-1)*pi*h/2.0)**2)
            cond = lmbd_max/lmbd_min
            tol = cond * self.eps

            n2 = n*n
            A = LinearOperator(n2, n2,
                               lambda x: Poisson2dMatvec(x),
                               symmetric=True)
            e = np.ones(n2)
            rhs = A * e
            cg = CG(A, matvec_max=2*n2, outputStream=sys.stderr)
            cg.solve(rhs)
            err = np.linalg.norm(e-cg.bestSolution)/n
            print(self.fmt % (n2, cg.nMatvec, cg.residNorm, err))

            # Adjust tol because allclose() uses infinity norm
            assert np.allclose(e, cg.bestSolution, rtol=err*n) == True
