import numpy as np


class PostProcessor:
    def __init__(self, result, n):
        self.result = result[0]
        self.n = n

        self.__r_part = None
        self.__i_part = None
        self.__l_ch_real = None
        self.__l_ch_imag = None

        self.l_ch = None
        self.density_matrix = None

    def __separate_result(self):
        sep = int((self.n ** 2 - self.n) / 2 + self.n)
        self.__r_part = self.result[:sep]
        self.__i_part = np.hstack((self.result[sep:], np.zeros(self.n)))

    @classmethod
    def __reshape_l_ch(cls, result_partition, n):
        l_ch = np.zeros((n, n))
        start_i = 0
        for i in range(n):
            l_ch += np.diag(result_partition[start_i:start_i + i + 1], -n + i + 1)
            start_i += i + 1
        return l_ch

    def __calculate_l_ch(self):
        self.__l_ch_real = self.__reshape_l_ch(self.__r_part, self.n)
        self.__l_ch_imag = self.__reshape_l_ch(self.__i_part, self.n) * 1j
        self.l_ch = self.__l_ch_real + self.__l_ch_imag

    def __calculate_density_matrix(self):
        self.density_matrix = self.l_ch.dot(self.l_ch.conj().transpose())

    def run(self):
        self.__separate_result()
        self.__calculate_l_ch()
        self.__calculate_density_matrix()
