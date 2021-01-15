from sqstate.preprocess import preprocess
from sqstate.model import get_model
from sqstate.postprocess import postprocess
from sqstate.utils import print_array


def main():
    data = preprocess("test_data_010701.mat")
    model = get_model("my_model_weights_0107.h5")
    result = model.predict(data)
    real, imag, dm = postprocess(result, 35)

    print("Real part:")
    print_array(real, [7, 28, 10, 25])

    print("Imag part:")
    print_array(imag, [7, 28, 10, 25])

    print("Density Matrix:")
    print_array(dm, [5, 32, 10, 25])
