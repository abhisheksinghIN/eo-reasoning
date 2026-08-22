"""Evidence-object construction."""

from __future__ import annotations

from models.evidence import EvidenceItem, EvidenceObject


def build_evidence_object(
    bbox: list,
    dates: list,
    spectral: dict,
    geofm: dict,
    physical_consistency: dict,
) -> dict:
    evidence = []

    for indicator in ["ndvi", "ndmi", "evi"]:
        stats = spectral[indicator]
        evidence.append(
            EvidenceItem(
                source=f"Sentinel-2/{indicator.upper()}",
                observation=(
                    f"{indicator.upper()} changed from "
                    f"{stats['start']:.4f} to {stats['end']:.4f}."
                ),
                value=float(stats["absolute_change"]),
                interpretation="Temporal spectral-indicator change.",
            )
        )

    geofm_change = geofm["summary"]["start_end_cosine_distance"]
    evidence.append(
        EvidenceItem(
            source="Prithvi-EO",
            observation=f"Start-to-end temporal embedding cosine distance is {geofm_change:.6f}.",
            value=float(geofm_change),
            interpretation="Change in learned GeoFM representation.",
        )
    )

    obj = EvidenceObject(
        task="vegetation_temporal_reasoning",
        aoi=bbox,
        dates=dates,
        observations={
            "n_frames": len(dates),
            "source": "Copernicus Data Space Ecosystem / Sentinel-2 L2A",
        },
        spectral=spectral,
        geofm={
            "model": geofm["model"],
            "input_shape": geofm["input_shape"],
            "summary": geofm["summary"],
        },
        physical_consistency=physical_consistency,
        evidence=evidence,
        provenance={
            "data_access": "CDSE STAC + Sentinel Hub Process API",
            "geofm": geofm["model"],
            "llm_role": "tool orchestration and evidence-grounded interpretation only",
        },
        limitations=[
            "Prithvi-EO-1.0-100M was pretrained on HLS data; Sentinel-2 L2A is used here as an MVP adaptation path.",
            "The current physical-consistency component is rule-based, not a learned physics-informed model.",
            "The current demo does not estimate causal drivers or soil moisture.",
        ],
    )
    return obj.model_dump()
