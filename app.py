"""Gradio interface for EO-Reasoning."""

from __future__ import annotations

import json

import gradio as gr

from agent.agent import EOAgent
from tools.cdse_tools import search_sentinel2_tool
from tools.pipeline import analyze_temporal_aoi


def _parse_bbox(text: str) -> list:
    values = [float(x.strip()) for x in text.split(",")]

    if len(values) != 4:
        raise ValueError(
            "BBox must be min_lon,min_lat,max_lon,max_lat"
        )

    return values


def ui_search(
    bbox_text,
    start_date,
    end_date,
    cloud_cover,
):
    try:
        return search_sentinel2_tool(
            bbox=_parse_bbox(bbox_text),
            start_date=start_date,
            end_date=end_date,
            max_cloud_cover=float(cloud_cover),
            limit=10,
        )

    except Exception as exc:
        return {
            "error": type(exc).__name__,
            "message": str(exc),
        }


def ui_analyze(
    bbox_text,
    date1,
    date2,
    date3,
):
    try:
        result = analyze_temporal_aoi(
            bbox=_parse_bbox(bbox_text),
            dates=[
                date1,
                date2,
                date3,
            ],
        )

        summary = result["physical_consistency"]

        ndvi_change = (
            result["spectral"]["ndvi"]["absolute_change"]
        )

        cosine_distance = (
            result["geofm"]["summary"][
                "start_end_cosine_distance"
            ]
        )

        markdown = (
            "### Analysis complete\n\n"
            f"- **NDVI change:** {ndvi_change:.4f}\n"
            f"- **Prithvi cosine distance:** "
            f"{cosine_distance:.6f}\n"
            f"- **Consistency:** "
            f"{summary['status']} "
            f"({summary['score']:.2f})\n\n"
            "> Prithvi cosine distance measures change in the "
            "learned GeoFM representation. It is not a direct "
            "physical measurement.\n\n"
            "> Current consistency is rule-based, not a "
            "calibrated physical model."
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
                "### Error\n"
                f"`{type(exc).__name__}: {exc}`"
            ),
            {
                "error": type(exc).__name__,
                "message": str(exc),
            },
            None,
            None,
        )


def ui_agent(question):
    try:
        result = EOAgent().run(question)

        trace = json.dumps(
            result["trace"],
            indent=2,
            default=str,
        )

        return (
            result["answer"],
            trace,
        )

    except Exception as exc:

        return (
            (
                f"Agent error: "
                f"{type(exc).__name__}: {exc}"
            ),
            "[]",
        )


with gr.Blocks(
    title="EO-Reasoning"
) as demo:

    gr.Markdown(
        """
# EO-Reasoning

### Evidence-grounded temporal Earth Observation reasoning with Prithvi

**CDSE → Data → GeoFM (Prithvi) → NDVI + latent change → evidence**

- The open-weight LLM is used only for **tool orchestration and interpretation**.
- NDVI provides an interpretable spectral vegetation-change baseline.
- Prithvi provides change in the learned GeoFM representation.
"""
    )

    default_bbox = (
        "11.25,46.40,11.40,46.55"
    )

    # -----------------------------------------------------
    # TAB 1 — DATA SEARCH
    # -----------------------------------------------------

    with gr.Tab("1 · Find data"):

        bbox = gr.Textbox(
            label="BBox",
            value=default_bbox,
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
                0,
                100,
                value=30,
                label="Max cloud cover (%)",
            )

        search_button = gr.Button(
            "Search CDSE"
        )

        search_output = gr.JSON(
            label="STAC results"
        )

        search_button.click(
            ui_search,
            inputs=[
                bbox,
                start,
                end,
                clouds,
            ],
            outputs=search_output,
        )

    # -----------------------------------------------------
    # TAB 2 — DETERMINISTIC ANALYSIS
    # -----------------------------------------------------

    with gr.Tab(
        "2 · Deterministic analysis"
    ):

        bbox2 = gr.Textbox(
            label="BBox",
            value=default_bbox,
        )

        gr.Markdown(
            """
Choose **three actual Sentinel-2 acquisition dates**
returned by the search tab.

The analysis produces:

- NDVI temporal change
- Prithvi latent representation change
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
            "Run GeoFM analysis"
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
                    "Download NDVI "
                    "Change GeoTIFF"
                )
            )

            prithvi_download = gr.File(
                label=(
                    "Download Prithvi "
                    "Change GeoTIFF"
                )
            )

        analyze_button.click(
            ui_analyze,
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

    # -----------------------------------------------------
    # TAB 3 — AGENT
    # -----------------------------------------------------

    with gr.Tab("3 · Agent"):

        gr.Markdown(
            "Requires Ollama and a "
            "tool-calling model such as Qwen3."
        )

        question = gr.Textbox(
            label="EO question",
            lines=4,
        )

        agent_button = gr.Button(
            "Ask EO-Reasoning"
        )

        answer = gr.Markdown(
            label="Answer"
        )

        trace = gr.Code(
            label="Tool trace",
            language="json",
        )

        agent_button.click(
            ui_agent,
            inputs=question,
            outputs=[
                answer,
                trace,
            ],
        )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
    )

#"""Gradio interface for EO-Reasoning."""
#
#from __future__ import annotations
#
#import json
#
#import gradio as gr
#
#from agent.agent import EOAgent
#from tools.cdse_tools import search_sentinel2_tool
#from tools.pipeline import analyze_temporal_aoi
#
#
#def _parse_bbox(text: str) -> list:
#    values = [float(x.strip()) for x in text.split(",")]
#    if len(values) != 4:
#        raise ValueError("BBox must be min_lon,min_lat,max_lon,max_lat")
#    return values
#
#
#def ui_search(bbox_text, start_date, end_date, cloud_cover):
#    try:
#        return search_sentinel2_tool(
#            bbox=_parse_bbox(bbox_text),
#            start_date=start_date,
#            end_date=end_date,
#            max_cloud_cover=float(cloud_cover),
#            limit=10,
#        )
#    except Exception as exc:
#        return {"error": type(exc).__name__, "message": str(exc)}
#
#
#def ui_analyze(bbox_text, date1, date2, date3):
#    try:
#        result = analyze_temporal_aoi(
#            bbox=_parse_bbox(bbox_text),
#            dates=[date1, date2, date3],
#        )
#        summary = result["physical_consistency"]
#        markdown = (
#            "### Analysis complete\n"
#            f"- **NDVI change:** {result['spectral']['ndvi']['absolute_change']:.4f}\n"
#            f"- **NDMI change:** {result['spectral']['ndmi']['absolute_change']:.4f}\n"
#            f"- **GeoFM cosine distance:** {result['geofm']['summary']['start_end_cosine_distance']:.6f}\n"
#            f"- **Consistency:** {summary['status']} ({summary['score']:.2f})\n\n"
#            "> Current consistency is rule-based, not a calibrated physical model."
#        )
#        return markdown, result
#    except Exception as exc:
#        return (
#            f"### Error\n`{type(exc).__name__}: {exc}`",
#            {"error": type(exc).__name__, "message": str(exc)},
#        )
#
#
#def ui_agent(question):
#    try:
#        result = EOAgent().run(question)
#        trace = json.dumps(result["trace"], indent=2, default=str)
#        return result["answer"], trace
#    except Exception as exc:
#        return f"Agent error: {type(exc).__name__}: {exc}", "[]"
#
#
#with gr.Blocks(title="EO-Reasoning") as demo:
#    gr.Markdown(
#        """
## EO-Reasoning
#### Evidence-grounded temporal Earth Observation reasoning with Prithvi
#
#**CDSE → Data → GeoFM (Prithvi) → spectral + embedding change → evidence**
#
#- The open-weight LLM is used only for **tool orchestration and interpretation**.
#"""
#    )
#
#    default_bbox = "11.25,46.40,11.40,46.55"
#
#    with gr.Tab("1 · Find data"):
#        bbox = gr.Textbox(label="BBox", value=default_bbox)
#        with gr.Row():
#            start = gr.Textbox(label="Start date", value="2026-06-01")
#            end = gr.Textbox(label="End date", value="2026-07-31")
#            clouds = gr.Slider(0, 100, value=30, label="Max cloud cover (%)")
#        search_button = gr.Button("Search CDSE")
#        search_output = gr.JSON(label="STAC results")
#        search_button.click(ui_search, inputs=[bbox, start, end, clouds], outputs=search_output)
#
#    with gr.Tab("2 · Deterministic analysis"):
#        bbox2 = gr.Textbox(label="BBox", value=default_bbox)
#        gr.Markdown("Choose **three actual acquisition dates** returned by the search tab.")
#        with gr.Row():
#            d1 = gr.Textbox(label="Date 1", value="2026-06-01")
#            d2 = gr.Textbox(label="Date 2", value="2026-06-15")
#            d3 = gr.Textbox(label="Date 3", value="2026-07-01")
#        analyze_button = gr.Button("Run GeoFM analysis")
#        analysis_summary = gr.Markdown()
#        evidence_json = gr.JSON(label="Evidence object")
#        analyze_button.click(
#            ui_analyze,
#            inputs=[bbox2, d1, d2, d3],
#            outputs=[analysis_summary, evidence_json],
#        )
#
#    with gr.Tab("3 · Agent"):
#        gr.Markdown("Requires Ollama and a tool-calling model such as Qwen3.")
#        question = gr.Textbox(label="EO question", lines=4)
#        agent_button = gr.Button("Ask EO-Reasoning")
#        answer = gr.Markdown(label="Answer")
#        trace = gr.Code(label="Tool trace", language="json")
#        agent_button.click(ui_agent, inputs=question, outputs=[answer, trace])
#
#
#if __name__ == "__main__":
#    demo.launch(server_name="0.0.0.0", server_port=7860)
