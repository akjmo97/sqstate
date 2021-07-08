import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))


from sqstate import CURRENT_PATH
from sqstate.preprocess import preprocess
from sqstate.model import get_model
from sqstate.postprocess import PostProcessor


def main():
    data_path = os.path.join(CURRENT_PATH, "../data")
    file_name = "test_sq_data.mat"

    n = 35
    model = get_model("Final_0321.h5")
    print(model.summary())

    # data = preprocess(os.path.join(data_path, file_name))
    # result = model.predict(data)

    # postprocessor = PostProcessor(result, n)
    # postprocessor.run()
    # l_ch, dm = postprocessor.l_ch, postprocessor.density_matrix

    # print(l_ch)
    # print(dm)


if __name__ == '__main__':
    main()
