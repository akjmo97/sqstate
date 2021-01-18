from sqstate.preprocess import preprocess
from sqstate.model import get_model
from sqstate.postprocess import PostProcessor
from sqstate.utils import ArrayPrinter
from sqstate.plot import calculate_wigner, plot_wigner


def main():
    n = 35
    data = preprocess("test_data_010701.mat")
    model = get_model("my_model_weights_0107.h5")
    result = model.predict(data)

    postprocessor = PostProcessor(result, n)
    postprocessor.run()
    l_ch, dm = postprocessor.l_ch, postprocessor.density_matrix

    print("L:")
    ArrayPrinter(l_ch, [5, 32, 10, 25]).print()

    print("Density Matrix:")
    ArrayPrinter(dm, [5, 32, 10, 25]).print()

    xs, ys, ws = calculate_wigner(
        dm,
        range(-int(n/2), int(n/2 + 1)),
        range(-int(n/2), int(n/2 + 1)),
        n
    )
    p = plot_wigner(xs, ys, ws)
    p.show()


if __name__ == '__main__':
    main()
