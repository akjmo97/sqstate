import numpy as np


def print_array(arr, boundary):
    space = 16*" " if isinstance(arr[0][0], np.complex128) else 7*" "
    for i, row in enumerate(arr):
        if boundary[2] < i < boundary[3]:
            if i == boundary[3] - 1:
                print("...")
            continue

        row_str = ""
        for j, n in enumerate(row):
            if boundary[0] < j < boundary[1]:
                row_str += "."
            elif n == 0:
                row_str += f"{space}0 "
            else:
                row_str += f"{n:+.1e} "
        print(row_str)
    print()
