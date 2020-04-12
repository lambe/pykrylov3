# -*- coding: utf-8 -*-
"""Limited-Memory SR1 (Symmetric Rank 1) Operators.

Linear operators to represent limited-memory SR1 matrices and their inverses.
L-SR1 matrices may not be positive-definite.
"""
from pykrylov3.linop.lqn.lqn import LQNLinearOperator, StructuredLQNLinearOperator

import numpy as np
from numpy.linalg import norm
from scipy.linalg.blas import ddot


class LSR1Operator(LQNLinearOperator):
    """Store and manipulate forward L-SR1 approximations.

    LSR1Operator may be used, e.g., in trust region methods, where the
    approximate Hessian is used in the model problem. L-SR1 has the advantage
    over L-BFGS and L-DFP of permitting approximations that are not
    positive-definite.

    This implementation uses an unrolling formula. The compact representation
    should be preferred if computational cost of the update is a concern.

    For this procedure see [Nocedal06].
    """

    def __init__(self, n, npairs=5, **kwargs):
        u"""Instantiate a :class: `LSR1Operator`.

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
        self.accept_threshold = 1e-8

    def _storing_test(self, new_s, new_y, ys):
        u"""Test if new pair {s, y} is to be stored.

        A new pair {s, y} is only accepted if

            ∣sᵀ(y - B s)∣ ⩾ 1e-8 ‖s‖ ‖y - B s‖.
        """
        Bs = self.qn_matvec(new_s)
        ymBs = new_y - Bs
        sTymBs = ddot(ymBs, new_s)
        norm_criterion = (abs(sTymBs) >=
            self.accept_threshold * norm(new_s) * norm(ymBs))
        sTymBs_criterion = abs(sTymBs) >= 1e-15

        ys_criterion = True
        scaling_criterion = True
        yms_criterion = True
        if self.scaling:
            ys_criterion = abs(ys) >= 1e-15
            scaling_factor = ys / ddot(new_y, new_y)
            scaling_criterion = (ys_criterion and
                                 norm(new_y - new_s / scaling_factor) >= 1e-10)
        else:
            yms_criterion = norm(new_y - new_s) >= 1e-10

        return (norm_criterion and
                sTymBs_criterion and
                yms_criterion and
                scaling_criterion and
                ys_criterion)

    def qn_matvec(self, v):
        """Compute matrix-vector product with L-SR1 approximation.

        Compute a matrix-vector product between the current limited-memory
        approximation to the Hessian matrix and the vector v using the
        unrolling formula.
        """
        self.n_matvec += 1

        q = v.copy()
        s = self.s
        y = self.y
        ys = self.ys
        paircount = len(s)
        a = [0.0] * paircount
        aTs = [0.0] * paircount

        if self.scaling and paircount > 0:
            self.gamma = ys[-1] / ddot(y[-1], y[-1])
            q /= self.gamma

        for i in range(paircount):
            a[i] = y[i] - s[i] / self.gamma
            for j in range(i):
                a[i] -= ddot(a[j], s[i]) / aTs[j] * a[j]
            aTs[i] = ddot(a[i], s[i])
            q += ddot(a[i], v) / aTs[i] * a[i]

        return q


class CompactLSR1Operator(LSR1Operator):
    """Store and manipulate forward L-SR1 approximations.

    The so-called compact representation is used to compute this approximation
    efficiently.
    """

    def __init__(self, n, npairs=5, **kwargs):
        u"""Instantiate a :class: `CompactLSR1Operator`.

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
        self.accept_threshold = 1.0e-8

    def qn_matvec(self, v):
        """Compute matrix-vector product with L-SR1 approximation.

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
        ys = self.ys
        paircount = len(s)
        a = [0.0] * paircount
        minimat = np.zeros([paircount, paircount], 'd')

        if self.scaling and paircount > 0:
            self.gamma = ys[-1] / ddot(y[-1], y[-1])
            q /= self.gamma

        for i in range(paircount):
            a[i] = ddot(y[i], v) - ddot(s[i], q)

        if paircount > 0:
            for i in range(paircount):
                minimat[i, i] = ys[i] - ddot(s[i], s[i]) / self.gamma
                for j in range(i):
                    minimat[i, j] = (ddot(s[i], y[j]) -
                                     ddot(s[i], s[j]) / self.gamma)
                    minimat[j, i] = minimat[i, j]

            b = np.linalg.solve(minimat, a)

            for i in range(paircount):
                q += b[i] * (y[i] - (1 / self.gamma) * s[i])
        return q


class InverseLSR1Operator(LSR1Operator):
    """Store and manipulate inverse L-SR1 approximations."""

    def __init__(self, n, npairs=5, **kwargs):
        u"""Instantiate a :class: `InverseLSR1Operator`.

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
        self.accept_threshold = 1e-8

    def qn_matvec(self, v):
        """Compute matrix-vector product with inverse L-SR1 approximation.

        Compute a matrix-vector product between the current limited-memory
        approximation to the inverse Hessian matrix and the vector v using
        the outer product representation.

        Note: there is probably some optimization that could be done in this
        function with respect to memory use and storing key dot products.
        """
        self.n_matvec += 1

        q = v.copy()
        s = self.s
        y = self.y
        ys = self.ys
        paircount = len(s)
        a = [0.0] * paircount
        minimat = np.zeros([paircount, paircount], 'd')

        if self.scaling and paircount > 0:
            self.gamma = ys[-1] / ddot(y[-1], y[-1])
            q *= self.gamma

        for i in range(paircount):
            a[i] = ddot(s[i], v) - ddot(y[i], q)

        if paircount > 0:
            for i in range(paircount):
                minimat[i, i] = ys[i] - ddot(y[i], y[i]) * self.gamma
                for j in range(i):
                    minimat[i, j] = (ddot(y[i], s[j]) -
                                     ddot(y[i], y[j]) * self.gamma)
                    minimat[j, i] = minimat[i, j]
            
            b = np.linalg.solve(minimat, a)

            for i in range(paircount):
                q += b[i] * (s[i] - self.gamma * y[i])

        return q


class StructuredLSR1Operator(StructuredLQNLinearOperator):
    """Store and manipulate structured forward L-SR1 approximations.

    Structured L-SR1 quasi-Newton approximation using an unrolling formula.
    For this procedure see [Nocedal06].
    """

    def __init__(self, n, npairs=5, **kwargs):
        u"""Instantiate a :class: `StructuredLSR1Operator`.

        :parameters:
            :n: the number of variables.
            :npairs: the number of {s, y, yd} pairs stored (default: 5).

        :keywords:
            :scaling: enable scaling of the 'initial matrix'. Scaling is
                         done as 'method M3' in the LBFGS paper by Zhou and
                         Nocedal; the scaling factor is sᵀy/yᵀy
                         (default: False).
        """
        super().__init__(n, npairs, **kwargs)
        self.accept_threshold = 1e-8

    def _storing_test(self, new_s, new_y, new_yd, ys):
        u"""Test if new pair {s, y, yd} is to be stored.

        A new pair {s, y, yd} is only accepted if

            ∣sᵀ(yd - B s)∣ ⩾ 1e-8 ‖s‖ ‖yd - B s‖.
        """
        Bs = self.qn_matvec(new_s)
        ymBs = new_yd - Bs
        sTymBs = ddot(ymBs, new_s)
        norm_criterion = (abs(sTymBs) >=
                          self.accept_threshold * norm(new_s) * norm(ymBs))
        sTymBs_criterion = abs(sTymBs) >= 1e-15

        ys_criterion = True
        scaling_criterion = True
        yms_criterion = True
        if self.scaling:
            ys_criterion = abs(ys) >= 1e-15
            scaling_factor = ys / ddot(new_y, new_y)
            scaling_criterion = norm(new_y - new_s / scaling_factor) >= 1e-10
        else:
            yms_criterion = norm(new_y - new_s) >= 1e-10

        return (norm_criterion and
                sTymBs_criterion and
                yms_criterion and
                scaling_criterion and
                ys_criterion)

    def qn_matvec(self, v):
        """Compute matrix-vector product with forward L-SR1 approximation.

        Compute a matrix-vector product between the current limited-memory
        approximation to the Hessian matrix and the vector v using
        the unrolling formula.
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
            # Form all a and ad vectors for the current step
            a[i] = y[i] - s[i] / self.gamma
            ad[i] = yd[i] - s[i] / self.gamma
            for j in range(i):
                aTs_i = ddot(a[j], s[i])
                adTs_i = ddot(ad[j], s[i])
                aTs_j = ddot(a[j], s[j])
                adTs_j = ddot(ad[j], s[j])
                update = ((aTs_i / aTs_j) * ad[j] +
                          (adTs_i / aTs_j) * a[j] -
                          (aTs_i * adTs_j / (aTs_j * aTs_j)) * a[j])
                a[i] -= update
                ad[i] -= update

            # Form inner products with current s and input vector
            aTs[i] = ddot(a[i], s[i])
            adTs[i] = ddot(ad[i], s[i])
            aTv = ddot(a[i], v)
            adTv = ddot(ad[i], v)
            q += ((aTv / aTs[i]) * ad[i] +
                  (adTv / aTs[i]) * a[i] -
                  (aTv * adTs[i] / (aTs[i] * aTs[i])) * a[i])

        return q
