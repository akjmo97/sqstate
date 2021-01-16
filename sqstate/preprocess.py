from sqstate import CURRENT_PATH
import os
import numpy as np
from scipy import io


def preprocess(file_name: str):
    data = io.loadmat(os.path.join(CURRENT_PATH, file_name))
    data = np.array(data['q_value'])
    data.astype(np.float32).reshape((1, 4032, 1))

    return data
