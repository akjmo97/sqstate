import numpy as np
from scipy.special import genlaguerre


def laguerre(x, n, alpha):
    # by Horner's method
    laguerre_l = 1
    b = 1
    for i in range(n, 0, -1):
        b *= (alpha + i)/(n + 1 - i)
        laguerre_l = b - x*laguerre_l/i

    return laguerre_l


def factorial_ij(i, j):
    ans = i
    while i <= j:
        ans *= i+1
        i += 1

    return ans


def moyal(x, p, m, n):
    if n >= m:
        w = 1 / np.pi
        w *= np.exp(-(np.square(x) + np.square(p))) * np.power(-1, m)

        w *= np.sqrt(np.power(2, n-m) / factorial_ij(m+1, n))
        w *= np.power(x - p*1j, n-m)
        w *= genlaguerre(m, n-m)(2*np.square(x) + 2*np.square(p))
        return w
    else:
        w = 1 / np.pi
        w *= np.exp(-(np.square(x) + np.square(p))) * np.power(-1, n)
        w *= np.sqrt(np.power(2, m-n) / factorial_ij(n+1, m))
        w *= np.power(x + p*1j, m-n)
        w *= genlaguerre(n, m-n)(2 * np.square(x) + 2 * np.square(p))
        return w


def wigner(dm: np.array, x_range: range):
    vec = np.array([i for i in x_range])
    xs, ys = np.meshgrid(vec, vec)

    w = np.zeros_like(dm)
    for i, row in enumerate(dm):
        for j, col in enumerate(row):
            w[i][j] += dm[i][j] * moyal(xs[i][j], ys[i][j], i, j)

    return xs, ys, w
