def laguerre(x, n, alpha):
    # by recurrence
    # if n == 0:
    #     return 1
    # elif n == 1:
    #     return -x + alpha + 1
    # elif n == 2:
    #     return (x**2 / 2) - ((alpha + 2) * x) + ((alpha + 2) * (alpha + 1) / 2)
    # else:
    #     return ((2*n + 1 + alpha - x) * laguerre(x, n-1, alpha) - (n + alpha) * laguerre(x, n-1, alpha)) / (n + 1)

    # by Horner's method
    laguerre_l = 1
    b = 1
    for i in range(n, 0, -1):
        b *= (alpha + i)/(n + 1 - i)
        laguerre_l = b - x*laguerre_l/i

    return laguerre_l
