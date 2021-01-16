from sqstate.main import get_1d_density_matrix


def test_get_1d_density_matrix():
    density_matrix_r, density_matrix_i = get_1d_density_matrix()
    assert density_matrix_r.shape == (1, 630)
    assert density_matrix_i.shape == (1, 630)
