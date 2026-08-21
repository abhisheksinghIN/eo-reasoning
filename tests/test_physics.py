from tools.physics_tools import check_vegetation_consistency


def test_consistent_case():
    result = check_vegetation_consistency(-0.1, -0.05, 0.2)
    assert result["status"] == "CONSISTENT"


def test_no_evidence_case():
    result = check_vegetation_consistency(0.1, 0.05, 0.0)
    assert result["status"] == "INSUFFICIENT"
