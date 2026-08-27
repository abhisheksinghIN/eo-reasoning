# EO-Reasoning

**Evidence-grounded temporal Earth Observation reasoning with Geo-Foundation Models and an agentic LLM**

> **Current MVP:** Sentinel-2 L2A → Prithvi-EO v1 → spectral indices + latent representation change → structured evidence → Qwen3 tool orchestration and interpretation.

> **How can Geo-Foundation Models enable deeper, evidence-grounded reasoning for Earth Observation?**

Most EO foundation-model demonstrations focus on downstream prediction or feature extraction. GeoReason-EO explores a different direction: using a GeoFM representation as one component in a transparent reasoning system that combines:

1. real EO observations;
2. deterministic spectral indicators;
3. learned GeoFM representations;
4. consistency checks;
5. provenance and limitations;
6. an LLM that orchestrates tools and explains evidence.
---

## 1. Project status

GeoReason-EO is a working research demonstrator for testing whether a pretrained geospatial foundation model can support **evidence-grounded temporal reasoning in Earth Observation (EO)**.

The current implementation has successfully demonstrated:

- Sentinel-2 data access through the **Copernicus Data Space Ecosystem (CDSE)**;
- Sentinel-2 L2A patch retrieval through the Sentinel Hub Process API;
- GPU inference with the TerraTorch Prithvi backbone (**Prithvi-EO v1**);
- temporal GeoFM embeddings with shape `[1, 3, 768]`;
- NDVI, NDMI, and EVI temporal statistics;
- latent-space cosine and L2 change metrics;
- structured evidence generation;
- Qwen3 tool calling through Ollama;
- evidence-grounded natural-language interpretation.

---

# 2. User interface

The Gradio application currently exposes three tabs.

## 2.1 Find data`

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
<img width="1023" height="539" alt="image" src="https://github.com/user-attachments/assets/39e52045-0d78-4469-a915-34f7b4ebed29" />

## 2.2 Deterministic analysis`

Purpose:

- accept exactly three acquisition dates;
- retrieve Sentinel-2 data;
- preprocess inputs;
- run Prithvi;
- calculate spectral indices;
- calculate embedding change;
- construct structured evidence.

The current Prithvi v1 MVP requires exactly three temporal frames.

<img width="1043" height="644" alt="image" src="https://github.com/user-attachments/assets/de6f40e8-b80c-4c45-84a8-d6323297fcdc" />

## 2.3 Agent`

Purpose:

- accept a natural-language EO request;
- let Qwen3 select high-level tools;
- execute deterministic EO analysis;
- present a tool trace;
- generate an evidence-grounded interpretation.

<img width="1040" height="497" alt="image" src="https://github.com/user-attachments/assets/d4ad3e66-a6cc-43ba-baea-a6d9adbd4508" />


---

# 3. System architecture

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
# 4. Repository structure

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
