import numpy as np
from scipy import io


def preprocess(file_path: str):
    data = io.loadmat(file_path)
    data = np.array(data['q_value'])
    data.astype(np.float32).reshape((1, 4032, 1))

    return data
