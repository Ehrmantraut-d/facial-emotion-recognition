import numpy as np
import pytest
from src.data.dataset import _parse_pixels

def test_parse_valid_pixels():
    img = _parse_pixels(" ".join(["128"] * 2304))
    assert img.shape == (48,48) and img.dtype == np.uint8 and img[0,0] == 128

@pytest.mark.parametrize("value", ["1 2 3", " ".join(["999"]*2304), " ".join(["nan"]*2304)])
def test_parse_rejects_invalid_pixels(value):
    with pytest.raises(ValueError): _parse_pixels(value)
