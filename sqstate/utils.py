import numpy as np


class ArrayPrinter:
    def __init__(self, arr, boundary=(-1, -1, -1, -1)):
        self.arr = arr
        self.boundary = boundary

    def __str__(self):
        space = 16*" " if isinstance(self.arr[0][0], np.complex128) else 7*" "

        arr_str = ""
        for i, row in enumerate(self.arr):
            if self.boundary[2] < i < self.boundary[3]:
                if i == self.boundary[3] - 1:
                    arr_str += "...\n"
                continue

            row_str = ""
            for j, n in enumerate(row):
                if self.boundary[0] < j < self.boundary[1]:
                    row_str += "."
                elif n == 0:
                    row_str += f"{space}0 "
                else:
                    row_str += f"{n:+.1e} "
            arr_str += f"{row_str}\n"

        return arr_str

    def print(self):
        print(self)
