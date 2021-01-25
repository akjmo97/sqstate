import numpy as np
from scipy.special import genlaguerre, factorial


def laguerre(x, n, alpha):
    # by Horner's method
    laguerre_l = 1
    b = 1
    for i in range(n, 0, -1):
        b *= (alpha + i) / (n + 1 - i)
        laguerre_l = b - x * laguerre_l / i

    return laguerre_l


def wigner(dm: np.array, x_range: range, g=0.5):
    vec = np.array([i for i in x_range])

    M = np.prod(dm.shape[0])
    X, Y = np.meshgrid(vec, vec)
    A = 0.5 * g * (X + 1.0j * Y)
    W = np.zeros(np.shape(A))

    B = 4 * abs(A) ** 2
    for m in range(M):
        if abs(dm[m, m]) > 0.0:
            W += np.real(dm[m, m] * (-1) ** m * genlaguerre(m, 0)(B))
        for n in range(m + 1, M):
            if abs(dm[m, n]) > 0.0:
                W += 2.0 * np.real(dm[m, n] * (-1) ** m *
                                   (2 * A) ** (n - m) *
                                   np.sqrt(factorial(m) / factorial(n)) *
                                   genlaguerre(m, n - m)(B))

    return X, Y, 0.5 * W * g ** 2 * np.exp(-B / 2) / np.pi
