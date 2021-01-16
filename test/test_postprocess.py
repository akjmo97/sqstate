import pytest
import numpy as np
from sqstate.postprocess import PostProcessor


class TestPostprocess:
    @pytest.fixture()
    def result_instance(self):
        return np.array([[1, 2, 3, 4, 5, 6, 1, 2, 3]])

    @pytest.fixture()
    def postprocessor_instance(self, result_instance):
        return PostProcessor(result_instance, 3)

    def test_separate_result(self, postprocessor_instance):
        postprocessor_instance._PostProcessor__separate_result()
        assert np.array_equal(postprocessor_instance._PostProcessor__r_part, np.array([1, 2, 3, 4, 5, 6]))
        assert np.array_equal(postprocessor_instance._PostProcessor__i_part, np.array([1, 2, 3, 0, 0, 0]))

    def test_reshape_l_ch(self, postprocessor_instance):
        postprocessor_instance._PostProcessor__separate_result()
        l_ch_real = postprocessor_instance._PostProcessor__reshape_l_ch(
            postprocessor_instance._PostProcessor__r_part,
            postprocessor_instance.n
        )
        l_ch_imag = postprocessor_instance._PostProcessor__reshape_l_ch(
            postprocessor_instance._PostProcessor__i_part,
            postprocessor_instance.n
        )
        assert np.array_equal(
            l_ch_real,
            np.array([
                [4, 0, 0],
                [2, 5, 0],
                [1, 3, 6],
            ])
        )
        assert np.array_equal(
            l_ch_imag,
            np.array([
                [0, 0, 0],
                [2, 0, 0],
                [1, 3, 0],
            ])
        )

    def test_calculate_l_ch(self, postprocessor_instance):
        postprocessor_instance._PostProcessor__separate_result()
        postprocessor_instance._PostProcessor__calculate_l_ch()
        assert np.array_equal(
            postprocessor_instance.l_ch,
            np.array([
                [4+0j, 0+0j, 0+0j],
                [2+2j, 5+0j, 0+0j],
                [1+1j, 3+3j, 6+0j],
            ])
        )

    def test_calculate_density_matrix(self, postprocessor_instance):
        postprocessor_instance._PostProcessor__separate_result()
        postprocessor_instance._PostProcessor__calculate_l_ch()
        postprocessor_instance._PostProcessor__calculate_density_matrix()
        l_ch = np.array([
            [4 + 0j, 0 + 0j, 0 + 0j],
            [2 + 2j, 5 + 0j, 0 + 0j],
            [1 + 1j, 3 + 3j, 6 + 0j],
        ])
        assert np.array_equal(
            postprocessor_instance.density_matrix,
            l_ch.dot(l_ch.conj().transpose())  # A = L L^*
        )

    def test_run(self, postprocessor_instance):
        postprocessor_instance.run()
        assert np.array_equal(
            postprocessor_instance.density_matrix,
            np.array([
                [16+0j, 8-8j, 4-4j],
                [8+8j, 33+0j, 19-15j],
                [4+4j, 19+15j, 56+0j],
            ])
        )
