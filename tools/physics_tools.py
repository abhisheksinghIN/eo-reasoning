"""Transparent consistency checks for the MVP.

This is rule-based, not a calibrated physics-informed model.
"""

from __future__ import annotations


def check_vegetation_consistency(
    ndvi_change: float,
    ndmi_change: float,
    embedding_cosine_distance: float,
) -> dict:
    evidence = []
    score = 0.0

    if ndvi_change < 0:
        evidence.append("NDVI decreased over the analysis period.")
        score += 0.35
    if ndmi_change < 0:
        evidence.append("NDMI decreased, consistent with reduced canopy/surface moisture.")
        score += 0.35
    if embedding_cosine_distance > 0:
        evidence.append("The GeoFM representation changed between the first and last frame.")
        score += 0.30

    if score >= 0.70:
        status = "CONSISTENT"
    elif score > 0:
        status = "PARTIAL"
    else:
        status = "INSUFFICIENT"

    return {
        "status": status,
        "score": round(min(score, 1.0), 3),
        "evidence": evidence,
        "method": "transparent_rule_based_consistency_check",
        "warning": (
            "This MVP check is not a calibrated physical model and must not be "
            "reported as a soil-moisture or vegetation-process estimate."
        ),
    }
