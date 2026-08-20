import gradio as gr


def health_check():
    return """
## GeoReason-EO

**Status:** Environment initialized successfully.

### Planned pipeline

CDSE STAC  
↓  
Sentinel-2 temporal observations  
↓  
Prithvi temporal GeoFM  
↓  
GeoFM embeddings  
↓  
Temporal change analysis  
↓  
NDVI / NDMI / EVI evidence  
↓  
Qwen3 tool-calling agent  
↓  
Evidence-grounded reasoning
"""


with gr.Blocks(
    title="GeoReason-EO"
) as demo:

    gr.Markdown(
        """
# GeoReason-EO

### Temporal Earth Observation Reasoning
**with Geospatial Foundation Models**

This is the initial project environment.
The scientific pipeline will be added incrementally.
"""
    )

    check = gr.Button(
        "Check system"
    )

    output = gr.Markdown()

    check.click(
        fn=health_check,
        outputs=output,
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
    )
