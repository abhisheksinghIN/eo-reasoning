# EO-Reasoning

**Evidence-grounded temporal Earth Observation reasoning with Geo-Foundation Models and an agentic LLM**

> **Current MVP:** Data Access→ Prithvi-EO v1 + TerraMind-v1 → latent representation → structured evidence → Qwen3 tool orchestration and interpretation.

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

<img width="432" height="119" alt="image" src="https://github.com/user-attachments/assets/8c00d31c-9b94-4ad0-be8f-8a9157236265" />

Input tensor: torch.Size([1, 3, 2, 224, 224]) | Backbone: terramind_v1_base | Modality: S1GRD | Device: cuda | Target date: 2026-07-17

**Metrics (Use-Case 2)**

Reference SSM: 60.5 | Reference noise: 7.5 | Observed VV median: 0.1092568039894104 | Incidence angle: 39.5174446105957

**Pre-Trained model (Terramind) outputs:**

Retrieval SSM: 53.05508804321289 | Predictive SSM: 58.102027893066406 | Predicted VV: 0.12611016631126404


**Losses:**

total        8.080045700073242 finite= True

retrieval-  6.944911956787109 finite= True | predictive - 1.8979721069335938 finite= True | jepa - 0.9307221174240112 finite= True | physics - 0.00014201791782397777 finite= True

<img width="407" height="665" alt="image" src="https://github.com/user-attachments/assets/0dde8e6a-5afb-4f92-bc34-a2c4c7b39287" />


The current implementation has successfully demonstrated:

- Data access through the **Copernicus Data Space Ecosystem (CDSE)**;
- Sentinel-1/2 patch retrieval through the Sentinel Hub Process API;
- GPU inference with the TerraTorch Prithvi backbone (**Prithvi-EO v1**, *TerraMind-v1*);
- structured evidence generation;
- Qwen3 tool calling through Ollama and evidence-grounded natural-language interpretation.

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

## 2.2 Deterministic analysis

Purpose:

- retrieve Sentinel-1/2 and reference data and preprocess inputs;
- run GeoFM and calculate embedding change;
- construct structured evidence.

<!--
<img width="755" height="399" alt="image" src="https://github.com/user-attachments/assets/21e15339-03ed-407e-839b-26c0a514b8aa" />
-->

## 2.3 Agent
e.g. (Use-case 1) Prompt: "Investigate vegetation temporal change for bbox [11.25, 46.40, 11.40, 46.55] between 2026-06-01 and 2026-07-31."

Purpose:

- accept a natural-language EO request;
- let Qwen3 select high-level tools;
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
