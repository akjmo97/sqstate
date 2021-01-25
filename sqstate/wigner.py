def laguerre(x, n, alpha):
    # by Horner's method
    laguerre_l = 1
    b = 1
    for i in range(n, 0, -1):
        b *= (alpha + i)/(n + 1 - i)
        laguerre_l = b - x*laguerre_l/i

    return laguerre_l
