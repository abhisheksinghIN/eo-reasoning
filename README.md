# EO-Reasoning

**Evidence-grounded temporal Earth Observation reasoning with Geo-Foundation Models and an agentic LLM**

> **Current MVP:** Data Access→ Prithvi-EO v1 + TerraMind-v1 → latent representation → structured evidence → ministral-3:8b tool orchestration and interpretation.

> **How can Geo-Foundation Models enable deeper, evidence-grounded reasoning for Earth Observation?**

Most EO foundation-model demonstrations focus on downstream prediction or feature extraction. GeoReason-EO explores a different direction: using a GeoFM representation as one component in a transparent reasoning system that combines:

1. real EO observations;
2. learned GeoFM representations;
3. an LLM that orchestrates tools and explains evidence.
---

## 1. Project status

**Use Case 1 – Change Detection:**
A research demonstrator designed to evaluate whether a pretrained geospatial foundation model can support **evidence-grounded temporal reasoning for Earth Observation (EO)** applications.

<img width="755" height="399" alt="image" src="https://github.com/user-attachments/assets/50b23471-fc55-48a9-b334-520781b7b450" />

**Use Case 2 – Soil Moisture Retrieval:**
A soil-moisture retrieval framework based on the **TerraMind geospatial foundation model (GeoFM)**, with a focus on **Sentinel-1 GRD data**, **physics-informed constraints**, and **JEPA-based learning**, integrated with the **Copernicus Data Space Ecosystem (CDSE)** through the **STAC API**.

<img width="491" height="554" alt="image" src="https://github.com/user-attachments/assets/68d68fd3-4e8b-406a-868b-6f66d6e7959f" />

<img width="504" height="484" alt="image" src="https://github.com/user-attachments/assets/4e7dc582-3f42-407f-83a5-c7a357ca732b" />

<img width="501" height="443" alt="image" src="https://github.com/user-attachments/assets/c1dd3a1e-424a-452b-a6c9-051783a85978" />


The current implementation has successfully demonstrated:

- Data access through the **Copernicus Data Space Ecosystem (CDSE)**;
- Sentinel-1/2 patch retrieval through the Sentinel Hub Process API;
- GPU inference with the TerraTorch Prithvi backbone (**Prithvi-EO v1**, *TerraMind-v1*);
- structured evidence generation;
- ministral-3:8b tool calling through Ollama and evidence-grounded natural-language interpretation.

---

# 2. User interface

The Gradio application currently exposes three tabs.

## 2.1 Find data

Purpose:

- search CDSE/STAC;
- inspect actual Sentinel-2 acquisitions;
- inspect acquisition dates;
- inspect metadata;
- inspect available assets.

Typical AOI:

```text
[11.25, 46.40, 11.40, 46.55]
```
<!--
<img width="1023" height="539" alt="image" src="https://github.com/user-attachments/assets/39e52045-0d78-4469-a915-34f7b4ebed29" />
-->

## 2.2 GeoFM Analysis (Vegetation and Soil Moisture)

Purpose:

- retrieve Sentinel-1/2 and reference data and preprocess inputs;
- run GeoFM and calculate embedding change;
- construct structured evidence.

<!--
<img width="755" height="399" alt="image" src="https://github.com/user-attachments/assets/21e15339-03ed-407e-839b-26c0a514b8aa" />
-->

## 2.3 Mistral Agent
e.g. (Use-case 1) Prompt: "Investigate vegetation temporal change for bbox [11.25, 46.40, 11.40, 46.55] between 2026-06-01 and 2026-07-31."

e.g. (Use-case 2) Prompt: "Analyze surface soil moisture for bbox [10.00, 45.05, 10.20, 45.20] in the Po Valley from 2026-07-01 to 2026-08-31. Use descending Sentinel-1 relative orbit 168. Explain the selected Sentinel-1 observations, the TerraMind GeoFM retrieval, the temporal JEPA prediction, SAR physics consistency, and the CLMS Surface Soil Moisture reference. Clearly distinguish observations from model outputs and reference data, and state all scientific limitations."

Purpose:

- accept a natural-language EO request;
- let mistral select high-level tools;
- execute deterministic EO analysis;
- generate an evidence-grounded interpretation.
<!--
<img width="1040" height="497" alt="image" src="https://github.com/user-attachments/assets/d4ad3e66-a6cc-43ba-baea-a6d9adbd4508" />
-->

---

# 3. System architecture
## 3.1 Use-Case 1

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
        Deterministic workflow                  ministral-3:8b / Ollama
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

                             ministral-3:8b interpretation
```
## 3.1 Use-Case 2

```text
                    
User prompt ─ Agent ┤ Mistral
                         │
                         ▼
              deterministic EO tools
                         │
                 Sentinel-1 search
                         │
             geometry/orbit selection
                         │
                     TerraMind
                         │
                  ┌──────┴──────┐
                  │             │
                 JEPA        retrieval
                  │             │
             predictive SSM     SSM
                  │             │
                  └─────┬───────┘
                        │
                     physics
                        │
                        ▼
               SoilMoistureEvidence
                        │
                        ▼
                     Mistral
                        │
               grounded explanation

```

The key design principle is:

> **Measurements come from deterministic EO tools. ministral-3:8b orchestrates tools and interprets returned evidence; it must not invent EO measurements.**

---
## 3.2 Repository structure 

The important logical structure is (Use-case 1):

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
