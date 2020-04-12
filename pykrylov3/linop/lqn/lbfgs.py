# -*- coding: utf-8 -*-
"""Limited-Memory BFGS Operators.

Linear operators to represent limited-memory BFGS matrices and their inverses.
"""

from pykrylov3.linop.lqn.lqn import LQNLinearOperator, StructuredLQNLinearOperator
import numpy as np
from scipy.linalg.blas import ddot

__docformat__ = 'restructuredtext'


class InverseLBFGSOperator(LQNLinearOperator):
    """Store and manipulate inverse L-BFGS approximations.

    :class: `InverseLBFGSOperator` may be used, e.g., in a L-BFGS solver for
    unconstrained minimization or as a preconditioner. The limited-memory
    matrix that is implicitly stored is a positive definite approximation to
    the inverse Hessian. Therefore, search directions may be obtained by
    computing matrix-vector products only. Such products are efficiently
    computed by means of a two-loop recursion.
    """

    def __init__(self, n, npairs=5, **kwargs):
        u"""Instantiate a :class: `InverseLBFGSOperator`.

        :parameters:
            :n: the number of variables.
            :npairs: the number of {s, y} pairs stored (default: 5).

        :keywords:
            :scaling: enable scaling of the 'initial matrix'. Scaling is
                         done as 'method M3' in the LBFGS paper by Zhou and
                         Nocedal; the scaling factor is sᵀy/yᵀy
                         (default: False).
        """
        super().__init__(n, npairs, **kwargs)

    def _storing_test(self, new_s, new_y, ys):
        u"""Test if new pair {s, y} is to be stored.

        A new pair is only accepted if the dot product yᵀs is over the
        threshold `self.accept_threshold`. The oldest pair is discarded in case
        the storage limit has been reached.
        """
        return ys > self.accept_threshold

    def qn_matvec(self, v):
        """Compute matrix-vector product with inverse L-BFGS approximation.

        Compute a matrix-vector product between the current limited-memory
        positive-definite approximation to the inverse Hessian matrix and the
        vector v using the L-BFGS two-loop recursion formula.
        """
        self.n_matvec += 1
        q = v.copy()
        s = self.s
        y = self.y
        ys = self.ys

        alpha = [0.0] * len(s)
        for k in range(len(s) - 1, -1, -1):
            alpha[k] = ddot(s[k], q) / ys[k]
            q -= alpha[k] * y[k]

        r = q
        if self.scaling and len(ys) > 0:
            self.gamma = ys[-1] / ddot(y[-1], y[-1])
            r *= self.gamma

        for k in range(len(s)):
            beta = ddot(y[k], r) / ys[k]
            r += (alpha[k] - beta) * s[k]

        return r


class LBFGSOperator(InverseLBFGSOperator):
    """Store and manipulate forward L-BFGS approximations.

    :class: `LBFGSOperator` is similar to :class: `InverseLBFGSOperator`,
    except that an approximation to the direct Hessian, not its inverse, is
    maintained.

    This form is useful in trust region methods, where the approximate Hessian
    is used in the model problem.
    """

    def __init__(self, n, npairs=5, **kwargs):
        u"""Instantiate a :class: `LBFGSOperator`.

        :parameters:
            :n: the number of variables.
            :npairs: the number of {s, y} pairs stored (default: 5).

        :keywords:
            :scaling: enable scaling of the 'initial matrix'. Scaling is
                         done as 'method M3' in the LBFGS paper by Zhou and
                         Nocedal; the scaling factor is sᵀy/yᵀy
                         (default: False).
        """
        super().__init__(n, npairs, **kwargs)

    def qn_matvec(self, v):
        """Compute matrix-vector product with forward L-BFGS approximation.

        Compute a matrix-vector product between the current limited-memory
        positive-definite approximation to the direct Hessian matrix and the
        vector v using the outer product representation.

        Note: there is probably some optimization that could be done in this
        function with respect to memory use and storing key dot products.
        """
        self.n_matvec += 1
        q = v.copy()
        s = self.s
        y = self.y
        ys = self.ys
        b = [0.0] * len(s)
        a = [0.0] * len(s)

        # B = Σ aa' - bb'.
        for k in range(len(s)):
            b[k] = y[k] / ys[k]**0.5
            q += ddot(b[k], v) * b[k]
            a[k] = s[k].copy()
            for j in range(k):
                a[k] += ddot(b[j], s[k]) * b[j]
                a[k] -= ddot(a[j], s[k]) * a[j]
            a[k] /= ddot(s[k], a[k])**0.5
            q -= ddot(a[k], v) * a[k]

        return q


class CompactLBFGSOperator(InverseLBFGSOperator):
    """Store and manipulate forward L-BFGS approximations in compact form.

    :class: `CompactLBFGSOperator` is similar to :class:
    `InverseLBFGSOperator`, except that it operates on the Hessian
    approximation directly, rather than the inverse. The so-called compact
    representation is used to compute this approximation efficiently.

    This form is useful in trust region methods, where the approximate Hessian
    is used in the model problem.
    """

    def __init__(self, n, npairs=5, **kwargs):
        u"""Instantiate a :class: `CompactLBFGSOperator`.

        :parameters:
            :n: the number of variables.
            :npairs: the number of {s, y} pairs stored (default: 5).

        :keywords:
            :scaling: enable scaling of the 'initial matrix'. Scaling is
                         done as 'method M3' in the LBFGS paper by Zhou and
                         Nocedal; the scaling factor is sᵀy/yᵀy
                         (default: False).
        """
        super().__init__(n, npairs, **kwargs)

    def qn_matvec(self, v):
        """Compute matrix-vector product with forward L-BFGS approximation.

        Compute a matrix-vector product between the current limited-memory
        positive-definite approximation to the direct Hessian matrix and the
        vector v using the outer product representation.

        Note: there is probably some optimization that could be done in this
        function with respect to memory use and caching key dot products.
        """
        self.n_matvec += 1

        q = v.copy()
        r = v.copy()
        s = self.s
        y = self.y
        ys = self.ys
        paircount = len(s)
        prodn = 2 * paircount
        a = np.zeros(prodn)
        minimat = np.zeros([prodn, prodn])

        if self.scaling and len(ys) > 0:
            self.gamma = ys[-1] / ddot(y[-1], y[-1])
            r *= self.gamma

        for i in range(paircount):
            a[i] = ddot(r, s[i])
            a[paircount + i] = ddot(q, y[i])

        if paircount > 0:
            for i in range(paircount):
                minimat[paircount + i, paircount + i] = -ys[i]
                minimat[i, i] = ddot(s[i], s[i]) / self.gamma
                for j in range(i):
                    minimat[i, paircount + j] = ddot(s[i], y[j])
                    minimat[paircount + j, i] = minimat[i, paircount + j]
                    minimat[i, j] = ddot(s[i], s[j]) / self.gamma
                    minimat[j, i] = minimat[i, j]

            b = np.linalg.solve(minimat, a)

            for i in range(paircount):
                r -= (b[i] / self.gamma) * s[i]
                r -= b[paircount + i] * y[i]

        return r


class StructuredLBFGSOperator(StructuredLQNLinearOperator):
    """Store and manipulate structured forward L-BFGS approximations.

    For this procedure see[Nocedal06].
    """

    def __init__(self, n, npairs=5, **kwargs):
        u"""Instantiate a :class: `StructuredLBFGSOperator`.

        :parameters:
            :n: the number of variables.
            :npairs: the number of {s,y, yd} pairs stored (default: 5).

        :keywords:
            :scaling: enable scaling of the 'initial matrix'. Scaling is
                         done as 'method M3' in the LBFGS paper by Zhou and
                         Nocedal; the scaling factor is sᵀy/yᵀy
                         (default: False).
        """
        super().__init__(n, npairs, **kwargs)
        self.accept_threshold = 1e-10

    def _storing_test(self, new_s, new_y, new_yd, ys):
        u"""Test if new pair {s, y, yd} is to be stored.

        A new pair {s, y, yd} is only accepted if

            ∣yᵀs + √(yᵀs sᵀBs)∣ ⩾ self.accept_threshold
        """
        Bs = self.qn_matvec(new_s)
        sBs = ddot(new_s, Bs)

        # Suppress python runtime warnings
        if ys < 0.0 or sBs < 0.0:
            return False

        ypBs = ys + (ys * sBs)**0.5
        return ypBs >= self.accept_threshold

    def qn_matvec(self, v):
        """Compute matrix-vector product with forward L-BFGS approximation.

        Compute a matrix-vector product between the current limited-memory
        approximation to the Hessian matrix and the vector v using
        the outer product representation.

        Note: there is probably some optimization that could be done in this
        function with respect to memory use and storing key dot products.
        """
        self.n_matvec += 1
        q = v.copy()
        s = self.s
        y = self.y
        yd = self.yd
        ys = self.ys
        paircount = len(s)
        a = [0.0] * paircount
        ad = [0.0] * paircount

        aTs = [0.0] * paircount
        adTs = [0.0] * paircount

        if self.scaling and paircount > 0:
            self.gamma = ys[-1] / ddot(y[-1], y[-1])
            q /= self.gamma

        for i in range(paircount):
            # Form a and ad vectors for current step
            ad[i] = yd[i] - s[i] / self.gamma
            Bs_i = s[i] / self.gamma
            for j in range(i):
                aTs_i = ddot(a[j], s[i])
                adTs_i = ddot(ad[j], s[i])
                aTs_j = ddot(a[j], s[j])
                adTs_j = ddot(a[j], s[j])
                update = ((aTs_i / aTs_j) * ad[j] +
                          (adTs_i / aTs_j) * a[j] -
                          (aTs_i * adTs_j / (aTs_j * aTs_j)) * a[j])
                Bs_i += update
                ad[i] -= update
            a[i] = y[i] + (ys[i] / ddot(s[i], Bs_i))**0.5 * Bs_i

            # Form inner products with current s and input vector
            aTs[i] = ddot(a[i], s[i])
            adTs[i] = ddot(ad[i], s[i])
            aTv = ddot(a[i], v)
            adTv = ddot(ad[i], v)
            q += ((aTv / aTs[i]) * ad[i] +
                  (adTv / aTs[i]) * a[i] -
                  (aTv * adTs[i] / (aTs[i] * aTs[i])) * a[i])

        return q
