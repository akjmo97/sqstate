from pytest import approx
from sqstate import wigner


def test_laguerre():
    tol = 1e-14

    x = 5
    alpha = 0.5
    assert approx(wigner.laguerre(x, 0, alpha), 1, abs=tol)
    assert approx(wigner.laguerre(x, 1, alpha), -x + alpha + 1, abs=tol)
    assert approx(
        wigner.laguerre(x, 2, alpha),
        (x ** 2) / 2 - ((alpha + 2) * x) + ((alpha + 2) * (alpha + 1)) / 2,
        abs=tol
    )
    assert approx(
        wigner.laguerre(x, 3, alpha),
        (-x ** 3) / 6 +
        ((alpha + 3) * x ** 2) / 2 -
        ((alpha + 2) * (alpha + 3) * x) / 2 +
        ((alpha + 1) * (alpha + 2) * (alpha + 3)) / 6,
        abs=tol
    )
