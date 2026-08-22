# GeoReason-EO

**Evidence-grounded temporal Earth Observation reasoning with Geo-Foundation Models and an agentic LLM**

> **Current MVP:** Sentinel-2 L2A → Prithvi-EO v1 → spectral indices + latent representation change → structured evidence → Qwen3 tool orchestration and interpretation.

---

## 1. Project status

GeoReason-EO is a working research demonstrator for testing whether a pretrained geospatial foundation model can support **evidence-grounded temporal reasoning in Earth Observation (EO)**.

The current implementation has successfully demonstrated:

- live Sentinel-2 discovery through the **Copernicus Data Space Ecosystem (CDSE)**;
- Sentinel-2 L2A patch retrieval through the Sentinel Hub Process API;
- a corrected nine-band Sentinel-2 stack;
- preprocessing into the six-channel, three-frame input expected by **Prithvi-EO v1**;
- GPU inference with the TerraTorch Prithvi backbone;
- temporal GeoFM embeddings with shape `[1, 3, 768]`;
- NDVI, NDMI, and EVI temporal statistics;
- latent-space cosine and L2 change metrics;
- structured evidence generation;
- a transparent rule-based consistency check;
- Qwen3 tool calling through Ollama;
- Qwen3 invocation of the deterministic `analyze_temporal_aoi` EO tool;
- evidence-grounded natural-language interpretation.

The main remaining work is **scientific hardening**, not basic infrastructure.

---

## 2. Research motivation

The broader research question is:

> **How can Geo-Foundation Models enable deeper, evidence-grounded reasoning for Earth Observation?**

Most EO foundation-model demonstrations focus on downstream prediction or feature extraction. GeoReason-EO explores a different direction: using a GeoFM representation as one component in a transparent reasoning system that combines:

1. real EO observations;
2. deterministic spectral indicators;
3. learned GeoFM representations;
4. consistency checks;
5. provenance and limitations;
6. an LLM that orchestrates tools and explains evidence.

The LLM is intentionally **not** treated as the numerical EO model.

---

## 3. Scientific scope of the current MVP

The current MVP supports:

**Primary task**

```text
vegetation_temporal_reasoning
```

**Current evidence sources**

- Sentinel-2 L2A surface reflectance;
- NDVI;
- NDMI;
- EVI;
- Prithvi-EO temporal embeddings;
- rule-based consistency logic.

**Not yet supported as scientific measurements**

- soil-moisture estimation;
- drought attribution;
- irrigation detection;
- vegetation-stress diagnosis;
- causal inference;
- calibrated process modelling;
- learned physics-informed inference;
- Sentinel-1 / SAR fusion;
- ERA5 forcing;
- multimodal GeoFM adaptation.

Those are later research stages.

---

# 4. System architecture

```text
                         ┌──────────────────────┐
                         │        User          │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │       Gradio         │
                         │   GeoReason-EO UI    │
                         └──────────┬───────────┘
                                    │
                  ┌─────────────────┴─────────────────┐
                  │                                   │
                  ▼                                   ▼
        Deterministic workflow                  Qwen3 / Ollama
                                                  orchestration
                  │                                   │
                  │                        ┌──────────┴──────────┐
                  │                        ▼                     ▼
                  │               search_sentinel2     analyze_temporal_aoi
                  │                        │                     │
                  └────────────────────────┴─────────────────────┘
                                           │
                                           ▼
                                  CDSE / Sentinel-2
                                           │
                                           ▼
                                  Process API retrieval
                                           │
                                           ▼
                                  Sentinel-2 preprocessing
                                           │
                                           ▼
                              [1, 6, 3, 224, 224]
                                           │
                                           ▼
                                     Prithvi-EO v1
                                           │
                                  ┌────────┴────────┐
                                  ▼                 ▼
                          spectral evidence    latent evidence
                         NDVI / NDMI / EVI     embeddings
                                  │                 │
                                  └────────┬────────┘
                                           ▼
                                consistency assessment
                                           │
                                           ▼
                                   EvidenceObject
                                           │
                                           ▼
                              Qwen3 interpretation
```

The key design principle is:

> **Measurements come from deterministic EO tools. Qwen3 orchestrates tools and interprets returned evidence; it must not invent EO measurements.**

---

# 5. User interface

The Gradio application currently exposes three tabs.

## 5.1 `1 · Find data`

Purpose:

- search CDSE/STAC;
- inspect actual Sentinel-2 acquisitions;
- inspect acquisition dates;
- inspect cloud-cover metadata;
- inspect available assets.

Typical AOI:

```text
[11.25, 46.40, 11.40, 46.55]
```

## 5.2 `2 · Deterministic analysis`

Purpose:

- accept exactly three acquisition dates;
- retrieve Sentinel-2 data;
- preprocess inputs;
- run Prithvi;
- calculate spectral indices;
- calculate embedding change;
- construct structured evidence.

The current Prithvi v1 MVP requires exactly three temporal frames.

## 5.3 `3 · Agent`

Purpose:

- accept a natural-language EO request;
- let Qwen3 select high-level tools;
- execute deterministic EO analysis;
- present a tool trace;
- generate an evidence-grounded interpretation.

---

# 6. Repository structure

The important logical structure is:

```text
eo-reasoning/
│
├── agent/
│   ├── agent.py
│   ├── prompts.py
│   └── tool_registry.py
│
├── analysis/
│   ├── temporal.py
│   ├── embedding_change.py
│   └── ...
│
├── data/
│   ├── cdse_auth.py
│   ├── stac_client.py
│   ├── sentinel2.py
│   └── preprocessing.py
│
├── models/
│   ├── prithvi.py
│   ├── temporal_encoder.py
│   ├── embeddings.py
│   └── evidence.py
│
├── tools/
│   ├── cdse_tools.py
│   ├── geofm_tools.py
│   ├── analysis_tools.py
│   ├── physics_tools.py
│   ├── evidence_tools.py
│   ├── agent_tools.py
│   └── pipeline.py
│
├── tests/
│   └── ...
│
├── app.py
├── requirements.txt
├── pyproject.toml
└── README.md
```
