import numpy as np
import matplotlib.pyplot as plt
from qutip.wigner import _wig_laguerre_val


def calculate_wigner(dm, x_range, y_range, n, scale=0.2):
    x_vec = [i for i in x_range]
    y_vec = [i for i in y_range]

    xs, ys = np.meshgrid(x_vec, y_vec)
    a2 = scale * (xs + 1.0j * ys)
    b = np.abs(a2) * np.abs(a2)

    w0 = 1
    while n > 0:
        n -= 1
        w0 = _wig_laguerre_val(n, b, np.diag(dm, n))

    return xs, ys, w0.real * np.exp(-b * 0.5) * (scale * scale * 0.5 / np.pi)


def plot_wigner(xs, ys, ws, title, z_lim=(-2e-2, 1e-2), size=(12, 9)):
    fig = plt.figure(figsize=size)
    ax = fig.gca(projection='3d')
    ax.plot_surface(xs, ys, ws, rstride=1, cstride=1, cmap="Spectral_r")
    ax.contourf(xs, ys, ws, 1000, zdir='z', offset=z_lim[0], cmap="GnBu")
    ax.contour(xs, ys, ws, zdir='z', offset=z_lim[0], colors='r', linewidths=0.5)
    # ax.clabel(c, fmt='%.2e', fontsize=8)
    ax.contourf(xs, ys, ws, 50, zdir='x', offset=xs[0, 0], cmap="winter")
    ax.contourf(xs, ys, ws, 50, zdir='y', offset=ys[-1, -1], cmap="winter")

    ax.set_zlim(z_lim[0], z_lim[1])
    ax.margins(0)
    ax.text2D(0.05, 0.95, title, transform=ax.transAxes)
    plt.tick_params(axis='both', labelbottom=False, labelleft=False)
    plt.tight_layout()

    return fig
