import os
from sqstate import CURRENT_PATH
from sqstate.preprocess import preprocess
from sqstate.model import get_model
from sqstate.postprocess import PostProcessor
from sqstate.utils import ArrayPrinter
from sqstate.wigner import wigner
from sqstate.plot import plot_wigner


def main():
    data_path = os.path.join(CURRENT_PATH, "../data")
    file_name = [
        f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))
    ]
    file_name.sort()

    n = 35
    model = get_model("my_model_weights_0107.h5")

    for file in file_name:
        data = preprocess(os.path.join(data_path, file))
        result = model.predict(data)

        postprocessor = PostProcessor(result, n)
        postprocessor.run()
        l_ch, dm = postprocessor.l_ch, postprocessor.density_matrix

        # print("L:")
        # ArrayPrinter(l_ch, [5, 32, 10, 25]).print()
        #
        # print("Density Matrix:")
        # ArrayPrinter(dm, [5, 32, 10, 25]).print()

        xs, ys, ws = wigner(dm, range(-int(n/2), int(n/2 + 1)))
        print(ws)
        # p = plot_wigner(xs, ys, ws, file)
        # p.show()


if __name__ == '__main__':
    main()
