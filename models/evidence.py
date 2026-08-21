"""Pydantic schemas for evidence-grounded EO reasoning."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    source: str
    observation: str
    value: Optional[float] = None
    unit: Optional[str] = None
    interpretation: Optional[str] = None


class EvidenceObject(BaseModel):
    task: str
    aoi: List[float]
    dates: List[str]
    observations: Dict[str, Any] = Field(default_factory=dict)
    spectral: Dict[str, Any] = Field(default_factory=dict)
    geofm: Dict[str, Any] = Field(default_factory=dict)
    physical_consistency: Dict[str, Any] = Field(default_factory=dict)
    evidence: List[EvidenceItem] = Field(default_factory=list)
    provenance: Dict[str, Any] = Field(default_factory=dict)
    limitations: List[str] = Field(default_factory=list)

    def as_llm_context(self) -> str:
        return self.model_dump_json(indent=2)
