# GeoReason-EO

## Temporal Earth Observation Reasoning with Geospatial Foundation Models

GeoReason-EO is an open research demonstrator for evidence-grounded
Earth Observation reasoning.

The system combines:

- Copernicus Earth Observation data
- Sentinel-2 temporal observations
- Geospatial Foundation Models (GeoFMs)
- Prithvi-EO
- temporal latent representations
- interpretable spectral indicators
- embedding-based change analysis
- open-weight Large Language Models
- tool-calling agents
- evidence-grounded scientific interpretation

The central research idea is to investigate whether temporal changes
in learned GeoFM representations can complement conventional
Earth Observation indicators for environmental interpretation.

---

## Architecture

```text
User
 │
 ▼
Gradio Interface
 │
 ▼
Open-Weight LLM
(Qwen3)
 │
 ├───────────────┐
 ▼               ▼
CDSE/STAC       GeoFM
 │              Prithvi
 ▼               │
Sentinel-2       ▼
time series   Temporal
 │            embeddings
 └───────┬───────────────┘
         ▼
Temporal Analysis
 │
 ├── NDVI
 ├── NDMI
 ├── EVI
 ├── temporal trends
 ├── embedding similarity
 └── embedding change
         │
         ▼
Evidence Object
         │
         ▼
LLM Scientific Interpretation

                   Qwen3
                     │
             ┌───────┴───────┐
             │   tool call   │
             ▼               │
        search_cdse          │
             │               │
             ▼               │
          CDSE/STAC          │
             │               │
             ▼               │
         Sentinel-2          │
             │               │
             ▼               │
          Prithvi            │
             │               │
             ▼               │
       temporal analysis     │
             │               │
             ▼               │
       physics checks        │
             │               │
             ▼               │
        EvidenceObject ──────┘
                     │
                     ▼
                 Qwen3
                     │
                     ▼
          scientific explanation
