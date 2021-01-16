from sqstate.preprocess import preprocess
from sqstate.model import get_model
from sqstate.postprocess import PostProcessor
from sqstate.utils import ArrayPrinter
import numpy as np
import matplotlib.pyplot as plt
from qutip.wigner import _wig_laguerre_val


def main():
    data = preprocess("test_data_010701.mat")
    model = get_model("my_model_weights_0107.h5")
    result = model.predict(data)

    postprocessor = PostProcessor(result, 35)
    postprocessor.run()
    l_ch, dm = postprocessor.l_ch, postprocessor.density_matrix

    print("L:")
    ArrayPrinter(l_ch, [5, 32, 10, 25]).print()

    print("Density Matrix:")
    ArrayPrinter(dm, [5, 32, 10, 25]).print()

    g = 0.2
    x_vec, y_vec = [i for i in range(-17, 18)], [i for i in range(-17, 18)]
    X, Y = np.meshgrid(x_vec, y_vec)
    A2 = g * (X + 1.0j * Y)
    B = np.abs(A2)
    B *= B
    L = 35
    w0 = 1
    while L > 0:
        L -= 1
        w0 = _wig_laguerre_val(L, B, np.diag(dm, L))

    w0 = w0.real * np.exp(-B*0.5) * (g*g*0.5 / np.pi)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, w0, rstride=1, cstride=1, cmap="Spectral_r")
    ax.contourf(X, Y, w0, 1000, zdir='z', offset=-2e-2, cmap="GnBu")
    ax.contour(X, Y, w0, zdir='z', offset=-2e-2, colors='r', linewidths=0.5)
    # ax.clabel(c, fmt='%.2e', fontsize=8)
    ax.contourf(X, Y, w0, 50, zdir='x', offset=-17, cmap="winter")
    ax.contourf(X, Y, w0, 50, zdir='y', offset=17, cmap="winter")

    ax.set_zlim(-2e-2, 1e-2)
    ax.margins(0)
    plt.tick_params(axis='both', labelbottom=False, labelleft=False)
    plt.tight_layout()

    plt.show()


if __name__ == '__main__':
    main()
