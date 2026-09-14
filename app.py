"""Gradio interface for EO-Reasoning."""

from __future__ import annotations

import json
import os

import gradio as gr

# ---------------------------------------------------------------------
# Existing vegetation / change-detection workflow
# ---------------------------------------------------------------------

from agents.agents import EOAgent
from tools.cdse_tools import search_sentinel2_tool
from tools.pipeline import analyze_temporal_aoi

# ---------------------------------------------------------------------
# New soil-moisture workflow
# ---------------------------------------------------------------------

from tools.soil_moisture_tool import (
    analyze_soil_moisture_aoi,
)

from agents.soil_moisture_agent import (
    DEFAULT_MODEL as DEFAULT_SOIL_AGENT_MODEL,
    run_agent as run_soil_moisture_agent,
)


# =====================================================================
# Shared helpers
# =====================================================================


def _parse_bbox(
    text: str,
) -> list[float]:
    """
    Parse:
        min_lon,min_lat,max_lon,max_lat
    """

    values = [
        float(x.strip())
        for x in text.split(",")
    ]

    if len(values) != 4:
        raise ValueError(
            "BBox must be "
            "min_lon,min_lat,max_lon,max_lat"
        )

    min_lon, min_lat, max_lon, max_lat = values

    if min_lon >= max_lon:
        raise ValueError(
            "BBox min_lon must be smaller than max_lon."
        )

    if min_lat >= max_lat:
        raise ValueError(
            "BBox min_lat must be smaller than max_lat."
        )

    return values


# =====================================================================
# VEGETATION / CHANGE-DETECTION UI FUNCTIONS
# =====================================================================


def ui_search(
    bbox_text,
    start_date,
    end_date,
    cloud_cover,
):
    """
    Search Sentinel-2 data through the existing CDSE tool.
    """

    try:

        return search_sentinel2_tool(
            bbox=_parse_bbox(
                bbox_text
            ),
            start_date=start_date,
            end_date=end_date,
            max_cloud_cover=float(
                cloud_cover
            ),
            limit=10,
        )

    except Exception as exc:

        return {
            "error": type(
                exc
            ).__name__,
            "message": str(
                exc
            ),
        }


def ui_analyze(
    bbox_text,
    date1,
    date2,
    date3,
):
    """
    Existing deterministic Sentinel-2 + Prithvi analysis.
    """

    try:

        result = analyze_temporal_aoi(
            bbox=_parse_bbox(
                bbox_text
            ),
            dates=[
                date1,
                date2,
                date3,
            ],
        )

        summary = result[
            "physical_consistency"
        ]

        ndvi_change = (
            result[
                "spectral"
            ][
                "ndvi"
            ][
                "absolute_change"
            ]
        )

        cosine_distance = (
            result[
                "geofm"
            ][
                "summary"
            ][
                "start_end_cosine_distance"
            ]
        )

        markdown = (
            "### Vegetation analysis complete\n\n"

            f"- **NDVI change:** "
            f"{ndvi_change:.4f}\n"

            f"- **Prithvi cosine distance:** "
            f"{cosine_distance:.6f}\n"

            f"- **Consistency:** "
            f"{summary['status']} "
            f"({summary['score']:.2f})\n\n"

            "> Prithvi cosine distance measures change "
            "in a learned GeoFM representation. "
            "It is not a direct physical measurement.\n\n"

            "> Current vegetation consistency logic is "
            "rule-based, not a calibrated physical model."
        )

        artifacts = result.get(
            "artifacts",
            {},
        )

        ndvi_file = artifacts.get(
            "ndvi_change_geotiff"
        )

        prithvi_file = artifacts.get(
            "prithvi_change_geotiff"
        )

        return (
            markdown,
            result,
            ndvi_file,
            prithvi_file,
        )

    except Exception as exc:

        return (
            (
                "### Error\n\n"
                f"`{type(exc).__name__}: {exc}`"
            ),
            {
                "error": type(
                    exc
                ).__name__,
                "message": str(
                    exc
                ),
            },
            None,
            None,
        )


def ui_vegetation_agent(
    question,
):
    """
    Existing EOAgent / vegetation reasoning interface.
    """

    try:

        result = EOAgent().run(
            question
        )

        trace = json.dumps(
            result["trace"],
            indent=2,
            default=str,
        )

        artifacts = result.get(
            "artifacts",
            {},
        )

        prithvi_map = artifacts.get(
            "prithvi_change_geotiff"
        )

        return (
            result["answer"],
            trace,
            prithvi_map,
        )

    except Exception as exc:

        return (
            (
                "Agent error: "
                f"{type(exc).__name__}: {exc}"
            ),
            "[]",
            None,
        )


# =====================================================================
# SOIL-MOISTURE UI FUNCTIONS
# =====================================================================


def ui_soil_moisture_analysis(
    bbox_text,
    start_date,
    end_date,
    orbit_state,
    relative_orbit,
):
    """
    Run the deterministic S1 + TerraMind + JEPA + physics tool.

    IMPORTANT:
    Current downstream SSM model is still an untrained prototype.
    """

    try:

        result = analyze_soil_moisture_aoi(
            bbox=_parse_bbox(
                bbox_text
            ),
            start_date=start_date,
            end_date=end_date,
            orbit_state=str(
                orbit_state
            ).lower(),
            relative_orbit=int(
                relative_orbit
            ),
        )

        if result.get(
            "status"
        ) != "success":

            message = result.get(
                "message",
                "Analysis could not be completed.",
            )

            markdown = (
                "### Soil-moisture analysis incomplete\n\n"
                f"{message}"
            )

            return (
                markdown,
                result,
            )

        model = result[
            "model"
        ]

        evidence = result[
            "model_evidence"
        ]

        reference = result[
            "reference"
        ]

        target = result[
            "target_observation"
        ]

        retrieval = evidence[
            "retrieval_ssm_percent"
        ]

        predictive = evidence[
            "predictive_ssm_percent"
        ]

        latent_distance = evidence[
            "jepa_latent_cosine_distance"
        ]

        physics_residual = evidence[
            "physics_vv_absolute_residual"
        ]

        reference_ssm = reference[
            "ssm_percent_saturation_median"
        ]

        reference_noise = reference[
            "ssm_noise_percent_median"
        ]

        markdown = (
            "### Soil-moisture prototype analysis complete\n\n"

            f"**Target date:** "
            f"{target['date']}\n\n"

            f"**Satellite:** "
            f"{target['platform']}\n\n"

            f"**GeoFM retrieval SSM:** "
            f"{retrieval:.2f}% saturation\n\n"

            f"**JEPA predictive SSM:** "
            f"{predictive:.2f}% saturation\n\n"

            f"**CLMS reference SSM:** "
            f"{reference_ssm:.2f}% saturation\n\n"

            f"**CLMS SSM noise:** "
            f"{reference_noise:.2f}%\n\n"

            f"**JEPA latent cosine distance:** "
            f"{latent_distance:.6f}\n\n"

            f"**Absolute SAR physics residual:** "
            f"{physics_residual:.6f}\n\n"

            "---\n\n"

            f"**Model status:** `{model['status']}`\n\n"

            "⚠️ **Scientific limitation:** "
            "TerraMind is pretrained, but the current "
            "soil-moisture head, JEPA predictor, and physics "
            "parameters have not yet been trained and validated. "
            "The retrieval and predictive SSM values above are "
            "**developmental prototype outputs**, not validated "
            "soil-moisture estimates.\n\n"

            "CLMS SSM is used as a reference / weak-supervision "
            "product and should not be described as independent "
            "ground truth."
        )

        return (
            markdown,
            result,
        )

    except Exception as exc:

        return (
            (
                "### Soil-moisture error\n\n"
                f"`{type(exc).__name__}: {exc}`"
            ),
            {
                "error": type(
                    exc
                ).__name__,
                "message": str(
                    exc
                ),
            },
        )


def ui_soil_moisture_agent(
    question,
    model_name,
):
    """
    Run local Mistral/Ollama orchestration.

    Detailed tool calls are also printed in the server terminal.
    """

    try:

        question = str(
            question
        ).strip()

        if not question:
            raise ValueError(
                "Please enter an EO question."
            )

        model_name = str(
            model_name
        ).strip()

        if not model_name:
            model_name = (
                DEFAULT_SOIL_AGENT_MODEL
            )

        answer = run_soil_moisture_agent(
            user_prompt=question,
            model_name=model_name,
            verbose=True,
        )

        status = (
            "### Agent execution complete\n\n"
            f"**LLM:** `{model_name}`\n\n"
            "The numerical EO evidence came from the "
            "deterministic soil-moisture tool. "
            "The LLM was used only for tool orchestration "
            "and evidence-grounded interpretation.\n\n"
            "Detailed tool-call arguments are also shown "
            "in the terminal running `app.py`."
        )

        return (
            answer,
            status,
        )

    except Exception as exc:

        return (
            (
                "### Agent error\n\n"
                f"`{type(exc).__name__}: {exc}`"
            ),
            (
                "### Execution failed\n\n"
                f"{type(exc).__name__}"
            ),
        )


# =====================================================================
# GRADIO APPLICATION
# =====================================================================


with gr.Blocks(
    title="EO-Reasoning"
) as demo:

    gr.Markdown(
        """
# EO-Reasoning

### GeoFoundation-model reasoning for Earth Observation

Current demonstrator:

**Use case 1 — Vegetation / change detection**

`Sentinel-2 → Prithvi → spectral + latent change → structured evidence`

**Use case 2 — Surface soil moisture**

`Sentinel-1 → TerraMind → temporal JEPA → SAR physics consistency → structured evidence`

The LLM is used only for **tool orchestration and interpretation**.
Quantitative EO values are generated by deterministic EO tools.

> **Soil-moisture status:** the current downstream SSM / JEPA / physics
> components are an **untrained research prototype**. Numerical SSM model
> outputs are not yet scientifically validated.
"""
    )

    # -----------------------------------------------------------------
    # Defaults
    # -----------------------------------------------------------------

    vegetation_bbox = (
        "11.25,46.40,11.40,46.55"
    )

    po_valley_bbox = (
        "10.00,45.05,10.20,45.20"
    )

    # =================================================================
    # TAB 1 — SENTINEL-2 SEARCH
    # =================================================================

    with gr.Tab(
        "1 · Vegetation · Find data"
    ):

        gr.Markdown(
            """
Search Sentinel-2 observations for the existing
vegetation / change-detection workflow.
"""
        )

        bbox = gr.Textbox(
            label="BBox",
            value=vegetation_bbox,
        )

        with gr.Row():

            start = gr.Textbox(
                label="Start date",
                value="2026-06-01",
            )

            end = gr.Textbox(
                label="End date",
                value="2026-08-31",
            )

            clouds = gr.Slider(
                minimum=0,
                maximum=100,
                value=30,
                label="Max cloud cover (%)",
            )

        search_button = gr.Button(
            "Search Sentinel-2"
        )

        search_output = gr.JSON(
            label="STAC results"
        )

        search_button.click(
            fn=ui_search,
            inputs=[
                bbox,
                start,
                end,
                clouds,
            ],
            outputs=[
                search_output,
            ],
        )

    # =================================================================
    # TAB 2 — VEGETATION ANALYSIS
    # =================================================================

    with gr.Tab(
        "2 · Vegetation · GeoFM analysis"
    ):

        bbox2 = gr.Textbox(
            label="BBox",
            value=vegetation_bbox,
        )

        gr.Markdown(
            """
Choose three actual Sentinel-2 acquisition dates
returned by the search tab.

Outputs include:

- NDVI temporal change
- Prithvi latent representation change
- rule-based consistency evidence
- downloadable GeoTIFF products
"""
        )

        with gr.Row():

            d1 = gr.Textbox(
                label="Date 1",
                value="2026-06-01",
            )

            d2 = gr.Textbox(
                label="Date 2",
                value="2026-07-16",
            )

            d3 = gr.Textbox(
                label="Date 3",
                value="2026-07-31",
            )

        analyze_button = gr.Button(
            "Run Prithvi analysis"
        )

        analysis_summary = gr.Markdown()

        evidence_json = gr.JSON(
            label="Evidence object"
        )

        gr.Markdown(
            "### Download change products"
        )

        with gr.Row():

            ndvi_download = gr.File(
                label=(
                    "NDVI Change GeoTIFF"
                )
            )

            prithvi_download = gr.File(
                label=(
                    "Prithvi Change GeoTIFF"
                )
            )

        analyze_button.click(
            fn=ui_analyze,
            inputs=[
                bbox2,
                d1,
                d2,
                d3,
            ],
            outputs=[
                analysis_summary,
                evidence_json,
                ndvi_download,
                prithvi_download,
            ],
        )

    # =================================================================
    # TAB 3 — EXISTING VEGETATION AGENT
    # =================================================================

    with gr.Tab(
        "3 · Vegetation · Agent"
    ):

        gr.Markdown(
            """
Existing vegetation/change-detection agent.

Requires Ollama and the model configured by the existing `EOAgent`.
"""
        )

        vegetation_question = gr.Textbox(
            label="EO question",
            lines=5,
            placeholder=(
                "Analyze vegetation change over "
                "the requested bbox and dates."
            ),
        )

        vegetation_agent_button = gr.Button(
            "Ask vegetation agent"
        )

        vegetation_answer = gr.Markdown(
            label="Answer"
        )

        vegetation_trace = gr.Code(
            label="Tool trace",
            language="json",
        )

        vegetation_prithvi_download = gr.File(
            label="Prithvi Change Map"
        )

        vegetation_agent_button.click(
            fn=ui_vegetation_agent,
            inputs=[
                vegetation_question,
            ],
            outputs=[
                vegetation_answer,
                vegetation_trace,
                vegetation_prithvi_download,
            ],
        )

    # =================================================================
    # TAB 4 — DETERMINISTIC SOIL-MOISTURE EVIDENCE
    # =================================================================

    with gr.Tab(
        "4 · Soil moisture · GeoFM evidence"
    ):

        gr.Markdown(
            """
Run the deterministic soil-moisture prototype directly,
without an LLM.

The pipeline selects compatible Sentinel-1 observations and runs:

**Sentinel-1 → TerraMind → JEPA → SSM head → SAR physics consistency**

CLMS Surface Soil Moisture is displayed separately as a
reference / weak-supervision product.
"""
        )

        soil_bbox = gr.Textbox(
            label="BBox",
            value=po_valley_bbox,
        )

        with gr.Row():

            soil_start = gr.Textbox(
                label="Start date",
                value="2026-07-01",
            )

            soil_end = gr.Textbox(
                label="End date",
                value="2026-08-31",
            )

        with gr.Row():

            soil_orbit_state = gr.Dropdown(
                choices=[
                    "descending",
                    "ascending",
                ],
                value="descending",
                label="Orbit direction",
            )

            soil_relative_orbit = gr.Number(
                value=168,
                precision=0,
                label="Relative orbit",
            )

        soil_analyze_button = gr.Button(
            "Run soil-moisture prototype",
            variant="primary",
        )

        soil_summary = gr.Markdown()

        soil_evidence = gr.JSON(
            label="Structured soil-moisture evidence"
        )

        soil_analyze_button.click(
            fn=ui_soil_moisture_analysis,
            inputs=[
                soil_bbox,
                soil_start,
                soil_end,
                soil_orbit_state,
                soil_relative_orbit,
            ],
            outputs=[
                soil_summary,
                soil_evidence,
            ],
        )

    # =================================================================
    # TAB 5 — MISTRAL SOIL-MOISTURE AGENT
    # =================================================================

    with gr.Tab(
        "5 · Soil moisture · Mistral agent"
    ):

        gr.Markdown(
            """
### Prompt-based EO reasoning

The local Mistral model does **not** calculate soil moisture.

It converts the prompt into a deterministic tool call and then
interprets the returned EO evidence.

The current numerical GeoFM soil-moisture outputs remain
**untrained prototype outputs**.
"""
        )

        soil_agent_model = gr.Textbox(
            label="Ollama model",
            value=DEFAULT_SOIL_AGENT_MODEL,
        )

        default_soil_prompt = (
            "Analyze surface soil moisture for bbox "
            "[10.00, 45.05, 10.20, 45.20] in the Po Valley "
            "from 2026-07-01 to 2026-08-31. "
            "Use descending Sentinel-1 relative orbit 168. "
            "Explain the selected Sentinel-1 observations, "
            "the TerraMind GeoFM retrieval, the temporal JEPA "
            "prediction, SAR physics consistency, and the CLMS "
            "Surface Soil Moisture reference. Clearly distinguish "
            "observations from model outputs and reference data, "
            "and state all scientific limitations."
        )

        soil_question = gr.Textbox(
            label="Soil-moisture question",
            value=default_soil_prompt,
            lines=8,
        )

        soil_agent_button = gr.Button(
            "Ask EO-Reasoning",
            variant="primary",
        )

        soil_agent_answer = gr.Markdown(
            label="Agent interpretation"
        )

        soil_agent_status = gr.Markdown(
            label="Execution status"
        )

        soil_agent_button.click(
            fn=ui_soil_moisture_agent,
            inputs=[
                soil_question,
                soil_agent_model,
            ],
            outputs=[
                soil_agent_answer,
                soil_agent_status,
            ],
        )


# =====================================================================
# LAUNCH
# =====================================================================


if __name__ == "__main__":

    server_name = os.getenv(
        "GRADIO_SERVER_NAME",
        "127.0.0.1",
    )

    server_port = int(
        os.getenv(
            "GRADIO_SERVER_PORT",
            "7860",
        )
    )

    share = (
        os.getenv(
            "GRADIO_SHARE",
            "false",
        ).lower()
        == "true"
    )

    # Queue requests so multiple GPU-heavy EO analyses
    # are not executed simultaneously.
    demo.queue()

    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        show_error=True,
    )


#"""Gradio interface for EO-Reasoning."""
#
#from __future__ import annotations
#
#import json
#
#import gradio as gr
#
#from agents.agent import EOAgent
#from tools.cdse_tools import search_sentinel2_tool
#from tools.pipeline import analyze_temporal_aoi
#
#
#def _parse_bbox(text: str) -> list:
#    values = [float(x.strip()) for x in text.split(",")]
#
#    if len(values) != 4:
#        raise ValueError(
#            "BBox must be min_lon,min_lat,max_lon,max_lat"
#        )
#
#    return values
#
#
#def ui_search(
#    bbox_text,
#    start_date,
#    end_date,
#    cloud_cover,
#):
#    try:
#        return search_sentinel2_tool(
#            bbox=_parse_bbox(bbox_text),
#            start_date=start_date,
#            end_date=end_date,
#            max_cloud_cover=float(cloud_cover),
#            limit=10,
#        )
#
#    except Exception as exc:
#        return {
#            "error": type(exc).__name__,
#            "message": str(exc),
#        }
#
#
#def ui_analyze(
#    bbox_text,
#    date1,
#    date2,
#    date3,
#):
#    try:
#        result = analyze_temporal_aoi(
#            bbox=_parse_bbox(bbox_text),
#            dates=[
#                date1,
#                date2,
#                date3,
#            ],
#        )
#
#        summary = result["physical_consistency"]
#
#        ndvi_change = (
#            result["spectral"]["ndvi"]["absolute_change"]
#        )
#
#        cosine_distance = (
#            result["geofm"]["summary"][
#                "start_end_cosine_distance"
#            ]
#        )
#
#        markdown = (
#            "### Analysis complete\n\n"
#            f"- **NDVI change:** {ndvi_change:.4f}\n"
#            f"- **Prithvi cosine distance:** "
#            f"{cosine_distance:.6f}\n"
#            f"- **Consistency:** "
#            f"{summary['status']} "
#            f"({summary['score']:.2f})\n\n"
#            "> Prithvi cosine distance measures change in the "
#            "learned GeoFM representation. It is not a direct "
#            "physical measurement.\n\n"
#            "> Current consistency is rule-based, not a "
#            "calibrated physical model."
#        )
#
#        artifacts = result.get(
#            "artifacts",
#            {},
#        )
#
#        ndvi_file = artifacts.get(
#            "ndvi_change_geotiff"
#        )
#
#        prithvi_file = artifacts.get(
#            "prithvi_change_geotiff"
#        )
#
#        return (
#            markdown,
#            result,
#            ndvi_file,
#            prithvi_file,
#        )
#
#    except Exception as exc:
#
#        return (
#            (
#                "### Error\n"
#                f"`{type(exc).__name__}: {exc}`"
#            ),
#            {
#                "error": type(exc).__name__,
#                "message": str(exc),
#            },
#            None,
#            None,
#        )
#
#
##def ui_agent(question):
##    try:
##        result = EOAgent().run(question)
##
##        trace = json.dumps(
##            result["trace"],
##            indent=2,
##            default=str,
##        )
##
##        return (
##            result["answer"],
##            trace,
##        )
##
##    except Exception as exc:
##
##        return (
##            (
##                f"Agent error: "
##                f"{type(exc).__name__}: {exc}"
##            ),
##            "[]",
##        )
#def ui_agent(question):
#    try:
#        result = EOAgent().run(question)
#
#        trace = json.dumps(
#            result["trace"],
#            indent=2,
#            default=str,
#        )
#
#        artifacts = result.get(
#            "artifacts",
#            {},
#        )
#
#        prithvi_map = artifacts.get(
#            "prithvi_change_geotiff"
#        )
#
#        return (
#            result["answer"],
#            trace,
#            prithvi_map,
#        )
#
#    except Exception as exc:
#        return (
#            (
#                f"Agent error: "
#                f"{type(exc).__name__}: {exc}"
#            ),
#            "[]",
#            None,
#        )
#
#with gr.Blocks(
#    title="EO-Reasoning"
#) as demo:
#
#    gr.Markdown(
#        """
## EO-Reasoning
#
#### Evidence-grounded temporal Earth Observation reasoning with Prithvi
#
#**CDSE → Data → GeoFM (Prithvi) → NDVI + latent change → evidence**
#
#- The open-weight LLM is used only for **tool orchestration and interpretation**.
#- NDVI provides an interpretable spectral vegetation-change baseline.
#- Prithvi provides change in the learned GeoFM representation.
#"""
#    )
#
#    default_bbox = (
#        "11.25,46.40,11.40,46.55"
#    )
#
#    # -----------------------------------------------------
#    # TAB 1 — DATA SEARCH
#    # -----------------------------------------------------
#
#    with gr.Tab("1 · Find data"):
#
#        bbox = gr.Textbox(
#            label="BBox",
#            value=default_bbox,
#        )
#
#        with gr.Row():
#
#            start = gr.Textbox(
#                label="Start date",
#                value="2026-06-01",
#            )
#
#            end = gr.Textbox(
#                label="End date",
#                value="2026-08-31",
#            )
#
#            clouds = gr.Slider(
#                0,
#                100,
#                value=30,
#                label="Max cloud cover (%)",
#            )
#
#        search_button = gr.Button(
#            "Search CDSE"
#        )
#
#        search_output = gr.JSON(
#            label="STAC results"
#        )
#
#        search_button.click(
#            ui_search,
#            inputs=[
#                bbox,
#                start,
#                end,
#                clouds,
#            ],
#            outputs=search_output,
#        )
#
#    # -----------------------------------------------------
#    # TAB 2 — DETERMINISTIC ANALYSIS
#    # -----------------------------------------------------
#
#    with gr.Tab(
#        "2 · Deterministic analysis"
#    ):
#
#        bbox2 = gr.Textbox(
#            label="BBox",
#            value=default_bbox,
#        )
#
#        gr.Markdown(
#            """
#Choose **three actual Sentinel-2 acquisition dates**
#returned by the search tab.
#
#The analysis produces:
#
#- NDVI temporal change
#- Prithvi latent representation change
#- downloadable GeoTIFF products
#"""
#        )
#
#        with gr.Row():
#
#            d1 = gr.Textbox(
#                label="Date 1",
#                value="2026-06-01",
#            )
#
#            d2 = gr.Textbox(
#                label="Date 2",
#                value="2026-07-16",
#            )
#
#            d3 = gr.Textbox(
#                label="Date 3",
#                value="2026-07-31",
#            )
#
#        analyze_button = gr.Button(
#            "Run GeoFM analysis"
#        )
#
#        analysis_summary = gr.Markdown()
#
#        evidence_json = gr.JSON(
#            label="Evidence object"
#        )
#
#        gr.Markdown(
#            "### Download change products"
#        )
#
#        with gr.Row():
#
#            ndvi_download = gr.File(
#                label=(
#                    "Download NDVI "
#                    "Change GeoTIFF"
#                )
#            )
#
#            prithvi_download = gr.File(
#                label=(
#                    "Download Prithvi "
#                    "Change GeoTIFF"
#                )
#            )
#
#        analyze_button.click(
#            ui_analyze,
#            inputs=[
#                bbox2,
#                d1,
#                d2,
#                d3,
#            ],
#            outputs=[
#                analysis_summary,
#                evidence_json,
#                ndvi_download,
#                prithvi_download,
#            ],
#        )
#
#
#    # -----------------------------------------------------
#    # TAB 3 — AGENT
#    # -----------------------------------------------------
#
#    with gr.Tab("3 · Agent"):
#
#        gr.Markdown(
#            "Requires Ollama and a "
#            "tool-calling model such as Qwen3."
#        )
#
#        question = gr.Textbox(
#            label="EO question",
#            lines=4,
#        )
#
#        agent_button = gr.Button(
#            "Ask EO-Reasoning"
#        )
#
#        answer = gr.Markdown(
#            label="Answer"
#        )
#
#        trace = gr.Code(
#            label="Tool trace",
#            language="json",
#        )
#
#        prithvi_download = gr.File(
#            label="Download Prithvi Change Map"
#        )
#
#        agent_button.click(
#            ui_agent,
#            inputs=question,
#            outputs=[
#                answer,
#                trace,
#                prithvi_download,
#            ],
#        )
#
#
#if __name__ == "__main__":
#    demo.launch(
#        server_name="127.0.0.1",
#        server_port=7860,
#    )
