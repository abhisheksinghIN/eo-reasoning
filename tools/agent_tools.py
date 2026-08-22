"""High-level tools exposed to the open-weight LLM."""

from tools.cdse_tools import search_sentinel2_tool
from tools.pipeline import analyze_temporal_aoi


def analyze_temporal_aoi_tool(bbox: list, dates: list) -> dict:
    """Analyze exactly three Sentinel-2 dates with Prithvi and return evidence."""
    return analyze_temporal_aoi(bbox=bbox, dates=dates)


TOOL_FUNCTIONS = {
    "search_sentinel2": search_sentinel2_tool,
    "analyze_temporal_aoi": analyze_temporal_aoi_tool,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_sentinel2",
            "description": "Search CDSE for Sentinel-2 L2A scenes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "bbox": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                    "max_cloud_cover": {"type": "number", "default": 30},
                    "limit": {"type": "integer", "default": 10},
                },
                "required": ["bbox", "start_date", "end_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_temporal_aoi",
            "description": (
                "Run CDSE retrieval, Prithvi embeddings, spectral indicators, "
                "temporal change, consistency checks and evidence construction."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "bbox": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                    "dates": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 3,
                        "maxItems": 3,
                    },
                },
                "required": ["bbox", "dates"],
            },
        },
    },
]
