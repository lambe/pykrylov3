"""Test LSR1 linear operators."""

import unittest
import numpy as np
from pykrylov3.linop.lqn import lsr1
from pykrylov3.tools import check_symmetric


class TestLSR1Operator(unittest.TestCase):
    """Test the various LSR1 linear operators."""

    def setUp(self):
        """Initialize."""
        self.n = 10
        self.npairs = 5
        self.B = lsr1.LSR1Operator(self.n, self.npairs)
        self.B_compact = lsr1.CompactLSR1Operator(self.n, self.npairs)
        self.H = lsr1.InverseLSR1Operator(self.n, self.npairs)

    def test_init(self):
        """Check that H = B = I initially."""
        rand_vec = np.random.random(self.n)
        assert np.allclose(self.B @ rand_vec, rand_vec)
        assert np.allclose(self.B_compact @ rand_vec, rand_vec)
        assert np.allclose(self.H @ rand_vec, rand_vec)

    def test_structure(self):
        """Test that B and H are inverses of each other."""
        # Insert a few {s,y} pairs.
        for _ in range(self.npairs + 2):
            s = np.random.random(self.n)
            y = np.random.random(self.n)
            self.B.store(s, y)
            self.B_compact.store(s, y)
            self.H.store(s, y)

        assert check_symmetric(self.B)
        assert check_symmetric(self.B_compact)
        assert check_symmetric(self.H)

        rand_vec = np.random.random(self.n)
        C = self.B @ self.H
        assert np.allclose(C @ rand_vec, rand_vec)
        C_compact = self.B_compact @ self.H
        assert np.allclose(C_compact @ rand_vec, rand_vec)
