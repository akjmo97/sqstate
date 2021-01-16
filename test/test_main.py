from sqstate.preprocess import preprocess
from sqstate.model import get_model
from sqstate.postprocess import PostProcessor
from sqstate.utils import ArrayPrinter
import numpy as np
import matplotlib.pyplot as plt
from qutip.wigner import wigner, _wig_laguerre_val


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

    g = 0.5
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

    plt.axes().contourf(x_vec, y_vec, w0, 100)
    plt.show()


if __name__ == '__main__':
    main()
