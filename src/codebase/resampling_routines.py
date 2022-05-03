import numpy as np
from numpy import random

"""
Copied from https://github.com/nchopin/particles/blob/5fa8d9135ecd7bf81721f07a4c6e57405e0a2cfd/particles/resampling.py

"""

def inverse_cdf(su, W):
    """Inverse CDF algorithm for a finite distribution.
        Parameters
        ----------
        su: (M,) ndarray
            M sorted uniform variates (i.e. M ordered points in [0,1]).
        W: (N,) ndarray
            a vector of N normalized weights (>=0 and sum to one)
        Returns
        -------
        A: (M,) ndarray
            a vector of M indices in range 0, ..., N-1
    """
    j = 0
    s = W[0]
    M = su.shape[0]
    A = np.empty(M, dtype=np.int64)
    for n in range(M):
        while su[n] > s:
            j += 1
            s += W[j]
        A[n] = j
    return A


def uniform_spacings(N):
    """ Generate ordered uniform variates in O(N) time.
    Parameters
    ----------
    N: int (>0)
        the expected number of uniform variates
    Returns
    -------
    (N,) float ndarray
        the N ordered variates (ascending order)
    Note
    ----
    This is equivalent to::
        from numpy import random
        u = sort(random.rand(N))
    but the line above has complexity O(N*log(N)), whereas the algorithm
    used here has complexity O(N).
    """
    z = np.cumsum(-np.log(random.rand(N + 1)))
    return z[:-1] / z[-1]


def multinomial_once(W):
    """ Sample once from a Multinomial distribution
    Parameters
    ----------
    W: (N,) ndarray
        normalized weights (>=0, sum to one)
    Returns
    -------
    int
        a single draw from the discrete distribution that generates n with
        probability W[n]
    Note
    ----
    This is equivalent to
       A = multinomial(W, M=1)
    but it is faster.
    """
    return np.searchsorted(np.cumsum(W), random.rand())


def multinomial(W, M):
    return inverse_cdf(uniform_spacings(M), W)

