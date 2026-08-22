import numpy as np

from data.preprocessing import ALL_BANDS, prepare_prithvi_frame, spectral_indices


def fake_frame():
    frame = np.zeros((len(ALL_BANDS), 8, 8), dtype=np.uint16)
    frame[0] = 1000
    frame[1] = 1200
    frame[2] = 1500
    frame[3] = 1800
    frame[4] = 2000
    frame[5] = 2200
    frame[6] = 3000
    frame[7] = 1800
    frame[8] = 4
    frame[9] = 1
    return frame


def test_prepare_prithvi_frame_shape():
    out = prepare_prithvi_frame(fake_frame())
    assert out.shape == (6, 8, 8)
    assert out.dtype == np.float32


def test_indices_are_finite():
    result = spectral_indices(fake_frame())
    assert np.isfinite(result["ndvi"])
    assert np.isfinite(result["ndmi"])
    assert np.isfinite(result["evi"])
    assert result["valid_fraction"] == 1.0
