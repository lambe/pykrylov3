import numpy as np
from scipy.linalg.blas import ddot, daxpy
from typing import Optional

from pykrylov3.tools import check_symmetric
from pykrylov3.generic import KrylovMethod

__docformat__ = 'restructuredtext'


class CG(KrylovMethod):
    """
    A pure Python implementation of the conjugate gradient (CG) algorithm. The
    conjugate gradient algorithm may be used to solve symmetric positive
    definite systems of linear equations, i.e., systems of the form

        A x = b

    where the operator A is square, symmetric and positive definite. This is
    equivalent to solving the unconstrained convex quadratic optimization
    problem

        minimize    -<b,x> + 1/2 <x, Ax>

    in the variable x.

    (CG may also be used to solve "quasi-definite" systems of linear equations,
    though the convergence properties change.)

    CG performs 1 operator-vector product, 2 dot products and 3 daxpys per
    iteration.

    If a preconditioner is supplied, it needs to solve one preconditioning
    system per iteration. Our implementation is standard and follows [Kelley]_
    and [Templates]_.
    """

    def __init__(self, op, **kwargs):
        super().__init__(op, **kwargs)

        self.name = 'Conjugate Gradient'
        self.acronym = 'CG'
        self.prefix = self.acronym + ': '

        # Direction of nonconvexity if A is not positive definite
        self.isPositiveDefinite: bool = True
        self.infiniteDescent: Optional[np.ndarray] = None

    def solve(self, rhs, x0=None, **kwargs):
        """
        Solve a linear system with `rhs` as right-hand side by the CG method.
        The vector `rhs` should be a Numpy array.

        :Keywords:

           :matvec_max:      Max. number of operator-vector produts. Default: 2n.
           :check_symmetric: Ensure operator is symmetric. Default: False.
           :check_curvature: Ensure operator is positive definite. Default: True.

        """
        n = rhs.shape[0]
        assert self.op.shape[0] == n

        self.nMatvec = 0
        check_sym = kwargs.get('check_symmetric', False)
        check_curvature = kwargs.get('check_curvature', True)

        if check_sym and not check_symmetric(self.op):
            self._writeerror('Coefficient operator is not symmetric')
            return

        # Check for an initial guess
        result_type = np.result_type(self.op.dtype, rhs.dtype)
        x = x0.copy() if x0 is not None else np.zeros(n, dtype=result_type)
        self._store_iterate(x)

        matvec_max = kwargs.get('matvec_max', 2*n)

        # Initial residual vector
        r = -rhs
        if x0 is not None:
            r += self.op * x
            self.nMatvec += 1

        # Initial preconditioned residual vector
        y = self.precon @ r
        self._store_resid(y)

        ry = ddot(r, y)
        self.residNorm0 = self.residNorm = ry**0.5
        self._store_resid_norm(self.residNorm0)
        threshold = max(self.abstol, self.reltol * self.residNorm0)

        p = -r   # Initial search direction (copy not to overwrite rhs if x=0)

        hdr_fmt = '%6s  %7s  %8s'
        hdr = hdr_fmt % ('Matvec', 'Resid', 'Curv')
        self._write(hdr)
        self._write('-' * len(hdr))
        info = '%6d  %7.1e' % (self.nMatvec, self.residNorm)
        self._write(info)

        while self.residNorm > threshold and self.nMatvec < matvec_max:
            Ap = self.op * p
            self.nMatvec += 1
            pAp = ddot(p, Ap)

            if check_curvature:
                if pAp <= 0:
                    self._writeerror('Coefficient operator is not positive definite')
                    self.infiniteDescent = p
                    self.isPositiveDefinite = False
                    break

            # Compute step length
            alpha = ry/pAp

            # Update estimate and residual
            x = daxpy(x, alpha * p)
            r = daxpy(r, alpha * Ap)

            self._store_iterate(x)

            # Compute preconditioned residual
            y = self.precon @ r
            self._store_resid(y)

            # Update preconditioned residual norm
            ry_next = ddot(r, y)

            # Update search direction
            beta = ry_next/ry
            p = daxpy(p, -r, a=beta)

            ry = ry_next
            self.residNorm = ry**0.5
            self._store_resid_norm(self.residNorm)

            info = '%6d  %7.1e  %8.1e' % (self.nMatvec, self.residNorm, pAp)
            self._write(info)

        self.converged = self.residNorm <= threshold
        self.bestSolution = x
