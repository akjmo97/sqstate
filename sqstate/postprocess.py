import numpy as np


def reshape_l_ch(result_partition, n):
    l_ch = np.zeros((n, n))
    start_i = 0
    for i in range(1, n):
        l_ch += np.diag(result_partition[start_i:start_i + i + 1], -n + i + 1)
        start_i += i+1

    return l_ch


def separate_result(result):
    r_part = result[:, :630][0]
    i_part = np.hstack((result[:, 630:], np.zeros((1, 35))))[0]

    return r_part, i_part


def combine_l_ch(r_part, i_part):
    return r_part + i_part * 1j


def calculate_density_matrix(l_ch):
    density_matrix = l_ch .dot(l_ch.conj().transpose())

    return density_matrix


def postprocess(result, n):
    r_part, i_part = separate_result(result)
    r_part, i_part = reshape_l_ch(r_part, n), reshape_l_ch(i_part, n)
    l_ch = combine_l_ch(r_part, i_part)
    density_matrix = calculate_density_matrix(l_ch)

    return r_part, i_part, density_matrix
