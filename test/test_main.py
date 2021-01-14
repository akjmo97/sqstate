from sqstate.main import mock_process


def test_mock_process():
    assert mock_process() == (1, 630)
