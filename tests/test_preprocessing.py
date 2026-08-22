import numpy as np

from data.preprocessing import (
    ALL_BANDS,
    prepare_prithvi_frame,
    spectral_indices,
)


def fake_frame():
    frame = np.zeros(
        (len(ALL_BANDS), 8, 8),
        dtype=np.uint16,
    )

    frame[0] = 1000   # B02
    frame[1] = 1200   # B03
    frame[2] = 1500   # B04
    frame[3] = 2800   # B8A
    frame[4] = 1800   # B11
    frame[5] = 1600   # B12
    frame[6] = 3000   # B08

    frame[7] = 4      # SCL: vegetation
    frame[8] = 1      # dataMask: valid

    return frame


def test_prepare_prithvi_frame_shape():
    out = prepare_prithvi_frame(fake_frame())

    assert out.shape == (6, 8, 8)
    assert np.isfinite(out).all()


def test_indices_are_finite():
    result = spectral_indices(fake_frame())

    assert np.isfinite(result["ndvi"])
    assert np.isfinite(result["ndmi"])
    assert np.isfinite(result["evi"])
    assert np.isfinite(result["valid_fraction"])

    assert result["valid_fraction"] == 1.0