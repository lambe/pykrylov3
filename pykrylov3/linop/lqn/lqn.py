# -*- coding: utf-8 -*-
"""Generic Limited-Memory Quasi Newton Operators.

Linear operators to represent limited-memory quasi-Newton matrices
or their inverses.
"""

import logging
from numpy import float
from scipy.linalg.blas import ddot
from scipy.sparse.linalg import LinearOperator

__docformat__ = 'restructuredtext'

# Default (null) logger.
null_log = logging.getLogger('lqn')
null_log.setLevel(logging.INFO)
null_log.addHandler(logging.NullHandler())


class LQNLinearOperator(LinearOperator):
    """Store and manipulate Limited-memory Quasi-Newton approximations."""

    def __init__(self, n, npairs=5, **kwargs):
        """Instantiate a :class: `LQNLinearOperator`.

        :parameters:
            :n: the number of variables.
            :npairs: the number of {s,y} pairs stored (default: 5).

        :keywords:
            :scaling: enable scaling of the 'initial matrix' (default: False).
        """
        # Mandatory arguments
        self._n = n
        self._npairs = npairs

        # Optional arguments
        self.scaling = kwargs.pop('scaling', False)

        # Threshold on dot product s'y to accept a new pair {s, y}.
        self.accept_threshold = 1.0e-20

        # Storage of the (s,y) pairs
        # Use a list structure where the newest vector is appended
        # to the end and the oldest is popped from the front on
        # calling the store() function
        self.s = list()     # np.empty((self.n, self.npairs), 'd')
        self.y = list()     # np.empty((self.n, self.npairs), 'd')
        self.ys = list()    # dot products si'yi
        self.gamma = 1.0

        # Keep track of number of matrix-vector products.
        self.n_matvec = 0

        # Assume dtype is standard double for now
        super().__init__(float, (n, n))

        self.logger = kwargs.get('logger', null_log)
        self.logger.info('New linear operator with shape ' + str(self.shape))

    @property
    def n(self):
        """The dimension of the (square) operator."""
        return self._n

    @property
    def npairs(self):
        """The maximum number of {s,y} pairs stored."""
        return self._npairs

    def _storing_test(self, new_s, new_y, ys):
        """Test if new pair {s, y} is to be stored."""
        raise NotImplementedError("Must be subclassed.")

    def store(self, new_s, new_y):
        """Store the new pair {new_s,new_y}.

        A new pair is only accepted if `self._storing_test()` is True. The
        oldest pair is then discarded in case the storage limit has been
        reached.
        """
        ys = ddot(new_s, new_y)

        if not self._storing_test(new_s, new_y, ys):
            self.logger.debug('Rejecting {s,y} pair')
            return

        self.s.append(new_s)
        self.y.append(new_y)
        self.ys.append(ys)

        if len(self.s) > self._npairs:
            _ = self.s.pop(0)
            _ = self.y.pop(0)
            _ = self.ys.pop(0)

        return

    def restart(self):
        """Restart the approximation by clearing all data on past updates."""
        self.ys = list()
        self.s = list()
        self.y = list()
        return

    def qn_matvec(self, v):
        """Compute matrix-vector product."""
        raise NotImplementedError("Must be subclassed.")

    def _matvec(self, x):
        return self.qn_matvec(x)


class StructuredLQNLinearOperator(LQNLinearOperator):
    u"""Store and manipulate structured limited-memory Quasi-Newton approximations.

    Structured quasi-Newton approximations may be used, e.g., in augmented Lagrangian methods or in nonlinear least-squares, where Hessian has a special structure.

    If Φ(x;λ,ρ) is the augmented Lagrangian function of an equality constrained optimization problem,
        ∇ₓₓΦ(x;λ,ρ) = ∇ₓₓL(x,λ+ρc(x)) + ρJ(x)ᵀJ(x).
    The structured quasi-Newton update takes the form
        B_{k+1} := S_{k+1} + ρJᵀ J
    where S_{k+1} ≈ ∇ₓₓL(x,λ+ρc(x)).
    See [Arreckx15]_ for more details.

    [Arreckx15] A matrix-free augmented lagrangian algorithm with application to large-scale structural design optimization, S. Arreckx, A. Lambe, J. R. R. A. Martins and D. Orban, Optimization and Engineering, 2015, 1--26.
    """

    def __init__(self, n, npairs=5, **kwargs):
        """Instantiate a :class: `StructuredLQNLinearOperator`.

        :parameters:
            :n: the number of variables.
            :npairs: the number of {s, y, yd} pairs stored (default: 5).

        :keywords:
            :scaling: enable scaling of the 'initial matrix' (default: False).
        """
        super().__init__(n, npairs, **kwargs)
        self.yd = list()    # np.empty((self.n, self.npairs))

    def store(self, new_s, new_y, new_yd):
        """Store the new pair {new_s, new_y, new_yd}.

        A new pair is only accepted if `self._storing_test()` is True. The
        oldest pair is then discarded in case the storage limit has been
        reached.
        """
        ys = ddot(new_s, new_y)

        if not self._storing_test(new_s, new_y, new_yd, ys):
            self.logger.debug('Rejecting {s, y, yd} set')
            return

        self.s.append(new_s)
        self.y.append(new_y)
        self.yd.append(new_yd)
        self.ys.append(ys)

        if len(self.s) > self._npairs:
            _ = self.s.pop(0)
            _ = self.y.pop(0)
            _ = self.yd.pop(0)
            _ = self.ys.pop(0)

        return

    def restart(self):
        super().restart()
        self.yd = list()
        return

    def _storing_test(self, new_s, new_y, new_yd, ys):
        """Test if new set {s, y, yd} is to be stored."""
        raise NotImplementedError("Must be subclassed.")
