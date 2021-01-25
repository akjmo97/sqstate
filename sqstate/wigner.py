import numpy as np
from scipy.special import factorial


def laguerre(x, n, alpha):
    # by Horner's method
    laguerre_l = 1
    b = 1
    for i in range(n, 0, -1):
        b *= (alpha + i)/(n + 1 - i)
        laguerre_l = b - x*laguerre_l/i

    return laguerre_l


def moyal(x, p, m, n):
    if n >= m:
        w = 1 / np.pi
        w *= np.exp(-(np.square(x) + np.square(p))) * np.power(-1, m)
        w *= np.sqrt(np.power(2, n-m) * factorial(m)/factorial(n))
        w *= np.power(x - p*1j, n-m)
        w *= laguerre((2*np.square(x) + 2*np.square(p)), m, n-m)
        return w
    else:
        w = 1 / np.pi
        w *= np.exp(-(np.square(x) + np.square(p))) * np.power(-1, n)
        w *= np.sqrt(np.power(2, m-n) * factorial(n) / factorial(m))
        w *= np.power(x + p*1j, m-n)
        w *= laguerre((2 * np.square(x) + 2 * np.square(p)), n, m-n)
        return w


def wigner(dm: np.array, x_range: range):
    vec = np.array([i for i in x_range])
    xs, ys = np.meshgrid(vec, vec)

    w = np.zeros_like(dm)
    for i, row in enumerate(dm):
        for j, col in enumerate(row):
            w[i][j] += dm[i][j] * moyal(xs[i][j], ys[i][j], i, j)

    return xs, ys, w
