from abc import ABC, abstractmethod
import logging
import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg.interface import IdentityOperator
from typing import List, Optional

__docformat__ = 'restructuredtext'

# Default (null) logger.
null_log = logging.getLogger('krylov')
null_log.setLevel(logging.INFO)
null_log.addHandler(logging.NullHandler())


class KrylovMethod(ABC):
    """
    A general template for implementing iterative Krylov methods. This module
    defines the `KrylovMethod` generic class. Other modules subclass
    `KrylovMethod` to implement specific algorithms.

    :parameters:

            :op:  an operator describing the coefficient matrix `A`.
                  `y = op * x` must return the operator-vector product
                  `y = Ax` for any given vector `x`. If required by
                  the method, `y = op.T * x` must return the operator-vector
                  product with the adjoint operator.

    :keywords:

        :atol:    absolute stopping tolerance. Default: 1.0e-8.

        :rtol:    relative stopping tolerance. Default: 1.0e-6.

        :precon:  optional preconditioner. If not `None`, `y = precon(x)`
                  returns the vector `y` solution of the linear system
                  `M y = x`.

        :logger:  a `logging.Logger` instance. If none is supplied, a default
                  null logger will be used.


    For general references on Krylov methods, see [Demmel]_, [Greenbaum]_,
    [Kelley]_, [Saad]_ and [Templates]_.

    References:

    .. [Demmel] J. W. Demmel, *Applied Numerical Linear Algebra*, SIAM,
                Philadelphia, 1997.

    .. [Greenbaum] A. Greenbaum, *Iterative Methods for Solving Linear Systems*,
                   number 17 in *Frontiers in Applied Mathematics*, SIAM,
                   Philadelphia, 1997.

    .. [Kelley] C. T. Kelley, *Iterative Methods for Linear and Nonlinear
                Equations*, number 16 in *Frontiers in Applied Mathematics*,
                SIAM, Philadelphia, 1995.

    .. [Saad] Y. Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed.,
              SIAM, Philadelphia, 2003.

    .. [Templates] R. Barrett, M. Berry, T. F. Chan, J. Demmel, J. M. Donato,
                   J. Dongarra, V. Eijkhout, R. Pozo, C. Romine and
                   H. Van der Vorst, *Templates for the Solution of Linear
                   Systems: Building Blocks for Iterative Methods*, SIAM,
                   Philadelphia, 1993.
    """

    def __init__(self, op: LinearOperator, **kwargs):

        self.prefix: str = 'Generic: '
        self.name: str = 'Generic Krylov Method (must be subclassed)'

        # Mandatory arguments
        self.op: LinearOperator = op

        # Optional keyword arguments
        self.abstol: float = kwargs.get('abstol', 1.0e-8)
        self.reltol: float = kwargs.get('reltol', 1.0e-6)
        self.precon: LinearOperator = kwargs.get('precon', IdentityOperator(op.shape, op.dtype))
        self.logger: logging.Logger = kwargs.get('logger', null_log)

        self.residNorm: Optional[np.float] = None
        self.residNorm0: Optional[np.float] = None

        self.storeResids: bool = kwargs.get('storeResids', False)
        self.residHistory: List[np.ndarray] = []
        self.storeResidNorms: bool = kwargs.get('storeResidNorms', False)
        self.residNormHistory: List[np.float] = []
        self.storeIterates: bool = kwargs.get('storeIterates', False)
        self.iterateHistory: List[np.ndarray] = []

        self.nMatvec: int = 0
        self.nIter: int = 0
        self.converged: bool = False
        self.bestSolution: Optional[np.ndarray] = None

    def _write(self, msg: str):
        self.logger.info(msg)

    def _writewarning(self, msg: str):
        self.logger.warning(msg)

    def _writeerror(self, msg: str):
        self.logger.error(msg)

    def _store_iterate(self, iterate: np.ndarray):
        if self.storeIterates:
            self.iterateHistory.append(iterate.copy())

    def _store_resid(self, resid: np.ndarray):
        if self.storeResids:
            self.residHistory.append(resid.copy())

    def _store_resid_norm(self, resid_norm: np.float):
        if self.storeResidNorms:
            self.residNormHistory.append(resid_norm)

    @abstractmethod
    def solve(self, rhs: np.ndarray, x0: Optional[np.ndarray] = None, **kwargs):
        """
        This is the :meth:`solve` method of the abstract `KrylovMethod` class.
        The class must be specialized and this method overridden.
        """
        raise NotImplementedError('This method must be subclassed')
