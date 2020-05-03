import numpy as np
from scipy.linalg.blas import ddot, daxpy

from pykrylov3.generic import KrylovMethod

__docformat__ = 'restructuredtext'


class CGS(KrylovMethod):
    """
    A pure Python implementation of the conjugate gradient squared (CGS)
    algorithm. CGS may be used to solve unsymmetric systems of linear equations,
    i.e., systems of the form

        A x = b

    where the operator A may be unsymmetric.

    CGS requires 2 operator-vector products with A, 3 dot products and 6 daxpys
    per iteration. It does not require products with the adjoint of A.

    If a preconditioner is supplied, CGS needs to solve two preconditioning
    systems per iteration. The original description appears in [Sonn89]_, which
    our implementation roughly follows.


    Reference:

    .. [Sonn89] P. Sonneveld, *CGS, A Fast Lanczos-Type Solver for Nonsymmetric
                Linear Systems*, SIAM Journal on Scientific and Statistical
                Computing **10** (1), pp. 36--52, 1989.
    """

    def __init__(self, op, **kwargs):
        KrylovMethod.__init__(self, op, **kwargs)

        self.name = 'Conjugate Gradient Squared'
        self.acronym = 'CGS'
        self.prefix = self.acronym + ': '

    def solve(self, rhs, x0=None, **kwargs):
        """
        Solve a linear system with `rhs` as right-hand side by the CGS method.
        The vector `rhs` should be a Numpy array.

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

        r0 = rhs  # Fixed vector throughout
        if x0 is not None:
            r0 -= self.op @ x
            self.nMatvec += 1
        self._store_resid(r0)

        rho = ddot(r0, r0)
        self.residNorm0 = self.residNorm = rho**0.5
        self._store_resid_norm()
        threshold = max(self.abstol, self.reltol * self.residNorm0)

        self._write('Initial residual = %8.2e\n' % self.residNorm0)
        self._write('Threshold = %8.2e\n' % threshold)

        finished = (self.residNorm <= threshold or self.nMatvec >= matvec_max)

        r = r0.copy()   # Initial residual vector
        u = r0.copy()
        p = r0.copy()

        while not finished:
            y = self.precon @ p
            v = self.op @ y
            self.nMatvec += 1

            sigma = ddot(r0, v)
            alpha = rho/sigma
            q = daxpy(v, u, a=-alpha)
            z = self.precon @ (u + q)

            # Update solution and residual
            x = daxpy(z, x, a=alpha)
            self._store_iterate(x)
            Az = self.op @ z
            self.nMatvec += 1
            r = daxpy(Az, r, a=-alpha)
            self._store_resid(r)

            # Update residual norm and check convergence
            self.residNorm = np.linalg.norm(r)
            self._store_resid_norm()
            finished = (self.residNorm <= threshold or self.nMatvec >= matvec_max)

            rho_next = ddot(r0, r)
            beta = rho_next/rho
            rho = rho_next
            u = daxpy(q, r, a=beta)

            p = daxpy(p, q, a=beta)
            p = daxpy(p, u, a=beta)

            # Display current info if requested
            self._write('%5d  %8.2e\n' % (self.nMatvec, self.residNorm))

        self.converged = self.residNorm <= threshold
        self.bestSolution = x
