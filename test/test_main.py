from sqstate.preprocess import preprocess
from sqstate.model import get_model
from sqstate.postprocess import postprocess
from sqstate.utils import ArrayPrinter


def main():
    data = preprocess("test_data_010701.mat")
    model = get_model("my_model_weights_0107.h5")
    result = model.predict(data)
    real, imag, dm = postprocess(result, 35)

    print("Real part:")
    ArrayPrinter(real, [7, 28, 10, 25]).print()

    print("Imag part:")
    ArrayPrinter(imag, [7, 28, 10, 25]).print()

    print("Density Matrix:")
    ArrayPrinter(dm, [5, 32, 10, 25]).print()


if __name__ == '__main__':
    main()
