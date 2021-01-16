import numpy as np
import sqstate.utils as utils


class TestUtils:
    def test_printer_real(self):
        printer = utils.ArrayPrinter(
            np.array([
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ])
        )
        assert str(printer) == \
            "+1.0e+00 +2.0e+00 +3.0e+00 \n" + \
            "+4.0e+00 +5.0e+00 +6.0e+00 \n" + \
            "+7.0e+00 +8.0e+00 +9.0e+00 \n"

    def test_printer_imag(self):
        printer = utils.ArrayPrinter(
            np.array([
                [1 + 1j, 2 + 2j, 3 + 3j],
                [4 + 4j, 5 + 5j, 6 + 6j],
                [7 + 7j, 8 + 8j, 9 + 9j],
            ])
        )
        assert str(printer) == \
            "+1.0e+00+1.0e+00j +2.0e+00+2.0e+00j +3.0e+00+3.0e+00j \n" + \
            "+4.0e+00+4.0e+00j +5.0e+00+5.0e+00j +6.0e+00+6.0e+00j \n" + \
            "+7.0e+00+7.0e+00j +8.0e+00+8.0e+00j +9.0e+00+9.0e+00j \n"
