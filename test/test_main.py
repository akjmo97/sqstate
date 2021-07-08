import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))


from sqstate import CURRENT_PATH
from sqstate.preprocess import preprocess
from sqstate.model import get_model
from sqstate.postprocess import PostProcessor
from sqstate.utils import ArrayPrinter
import h5py


def main():
    data_path = os.path.join(CURRENT_PATH, "../data")
    file_name = [
        # f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))
        "SQ4"
    ]
    file_name.sort()

    n = 35
    model = get_model("my_model_weights_0107.h5")
    # model = get_model("Final_0321.h5")

    # for file in file_name:
    #     data = preprocess(os.path.join(data_path, file))
    #     result = model.predict(data)

    #     postprocessor = PostProcessor(result, n)
    #     postprocessor.run()
    #     l_ch, dm = postprocessor.l_ch, postprocessor.density_matrix

        # f = h5py.File(os.path.join(CURRENT_PATH, "../data/dm.hdf5"), "w")
        # f.create_group(file)
        # f.create_dataset(f"{file}/real", data=dm.real)
        # f.create_dataset(f"{file}/imag", data=dm.imag)
        # f.close()

        # print("Real part")
        # print(dm.real[0])
        # print("Imag part")
        # print(dm.imag[0])

        # f = h5py.File(os.path.join(CURRENT_PATH, "../data/dm.hdf5"), "r")
        # print("Real part")
        # print(f.get(f"{file}/real")[()][0])
        # print("Imag part")
        # print(f.get(f"{file}/imag")[()][0])
        # f.close()

        # print("L:")
        # ArrayPrinter(l_ch, [5, 32, 10, 25]).print()
        #
        # print("Density Matrix:")
        # ArrayPrinter(dm, [5, 32, 10, 25]).print()


if __name__ == '__main__':
    main()
