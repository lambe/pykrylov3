import numpy as np
from scipy.linalg.blas import ddot, daxpy

from pykrylov3.generic import KrylovMethod

__docformat__ = 'restructuredtext'


class BiCGSTAB(KrylovMethod):
    """
    A pure Python implementation of the bi-conjugate gradient stabilized
    (Bi-CGSTAB) algorithm. Bi-CGSTAB may be used to solve unsymmetric systems
    of linear equations, i.e., systems of the form

        A x = b

    where the operator A is unsymmetric and nonsingular.

    Bi-CGSTAB requires 2 operator-vector products, 6 dot products and 6 daxpys
    per iteration.

    In addition, if a preconditioner is supplied, it needs to solve 2
    preconditioning systems per iteration.

    The original description appears in [VdVorst92]_. Our implementation is a
    preconditioned version of that given in [Kelley]_.

    Reference:

    .. [VdVorst92] H. Van der Vorst, *Bi-CGSTAB: A Fast and Smoothly Convergent
                   Variant of Bi-CG for the Solution of Nonsymmetric Linear
                   Systems*, SIAM Journal on Scientific and Statistical
                   Computing **13** (2), pp. 631--644, 1992.
    """

    def __init__(self, op, **kwargs):
        KrylovMethod.__init__(self, op, **kwargs)

        self.name = 'Bi-Conjugate Gradient Stabilized'
        self.acronym = 'Bi-CGSTAB'
        self.prefix = self.acronym + ': '

    def solve(self, rhs, x0=None, **kwargs):
        """
        Solve a linear system with `rhs` as right-hand side by the Bi-CGSTAB
        method. The vector `rhs` should be a Numpy array.

        :keywords:
            :matvec_max: Max. number of matrix-vector produts (2n)
        """
        n = rhs.shape[0]
        assert self.op.shape[0] == n
        self.nMatvec = 0

        # Initial guess is zero unless one is supplied
        result_type = np.result_type(self.op.dtype, rhs.dtype)
        x = x0.copy() if x0 is not None else np.zeros(n, dtype=result_type)
        self._store_iterate(x)

        matvec_max = kwargs.get('matvec_max', 2*n)

        # Initial residual is the fixed vector
        r0 = rhs
        if x0 is not None:
            r0 -= self.op @ x
            self.nMatvec += 1

        rho = alpha = omega = 1.0
        rho_next = ddot(r0, r0)
        self.residNorm = self.residNorm0 = rho_next**0.5
        self._store_resid_norm()
        threshold = max(self.abstol, self.reltol * self.residNorm0)

        self._write('Initial residual = %8.2e' % self.residNorm0)
        self._write('Threshold = %8.2e' % threshold)
        hdr = '%6s  %8s' % ('Matvec', 'Residual')
        self._write(hdr)
        self._write('-' * len(hdr))

        finished = (self.residNorm <= threshold or self.nMatvec >= matvec_max)

        r = r0.copy()
        p = np.zeros(n, dtype=result_type)
        v = np.zeros(n, dtype=result_type)

        while not finished:
            beta = rho_next/rho * alpha/omega
            rho = rho_next

            p = daxpy(v, p, a=omega)
            p = daxpy(p, r, a=beta)

            # Compute preconditioned search direction
            q = self.precon @ p
            v = self.op @ q
            self.nMatvec += 1

            alpha = rho / ddot(r0, v)
            s = daxpy(v, r, a=-alpha)

            # Check for CGS termination
            cgs_residNorm = np.linalg.norm(s)
            self._write('%6d  %8.2e' % (self.nMatvec, self.residNorm))

            if cgs_residNorm <= threshold:
                x = daxpy(q, x, a=alpha)
                break

            z = self.precon @ s
            t = self.op @ z
            self.nMatvec += 1

            omega = ddot(t, s) / ddot(t, t)
            rho_next = -omega * ddot(r0, t)

            # Update residual
            r = daxpy(t, s, a=-omega)
            self._store_resid(r)

            # Update solution
            x = daxpy(z, x, a=omega)
            x = daxpy(q, x, a=alpha)

            self.residNorm = np.linalg.norm(r)
            self._store_resid_norm()

            finished = (self.residNorm <= threshold or self.nMatvec >= matvec_max)
            self._write('%6d  %8.2e' % (self.nMatvec, self.residNorm))

        self.converged = self.residNorm <= threshold
        self.bestSolution = x
