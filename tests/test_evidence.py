from tools.evidence_tools import build_evidence_object


def test_evidence_object():
    spectral = {
        "ndvi": {"start": 0.7, "end": 0.5, "absolute_change": -0.2},
        "ndmi": {"start": 0.4, "end": 0.3, "absolute_change": -0.1},
        "evi": {"start": 0.5, "end": 0.4, "absolute_change": -0.1},
    }
    geofm = {
        "model": "test-model",
        "input_shape": [1, 6, 3, 224, 224],
        "summary": {"start_end_cosine_distance": 0.2},
    }
    result = build_evidence_object(
        bbox=[10, 46, 11, 47],
        dates=["2026-06-01", "2026-06-15", "2026-07-01"],
        spectral=spectral,
        geofm=geofm,
        physical_consistency={"status": "CONSISTENT"},
    )
    assert result["task"] == "vegetation_temporal_reasoning"
    assert len(result["evidence"]) == 4
