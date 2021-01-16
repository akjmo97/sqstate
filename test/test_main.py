from sqstate.preprocess import preprocess
from sqstate.model import get_model
from sqstate.postprocess import PostProcessor
from sqstate.utils import ArrayPrinter


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


if __name__ == '__main__':
    main()
