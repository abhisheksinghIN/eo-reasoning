"""Ollama/Qwen tool-calling loop for EO reasoning."""

from __future__ import annotations

import json
import os
from datetime import date

from ollama import Client

from agent.prompts import SYSTEM_PROMPT
from agent.tool_registry import (
    TOOL_FUNCTIONS,
    TOOL_SCHEMAS,
)


def _select_three_observations(items: list[dict]) -> list[str]:
    """
    Select three real Sentinel-2 acquisition dates with
    temporal spread.

    Strategy:
        earliest
        middle
        latest
    """

    valid_items = [
        item
        for item in items
        if item.get("date")
    ]

    if len(valid_items) < 3:
        raise ValueError(
            f"Need at least 3 Sentinel-2 observations; "
            f"found {len(valid_items)}."
        )

    valid_items = sorted(
        valid_items,
        key=lambda item: date.fromisoformat(item["date"]),
    )

    first = valid_items[0]
    last = valid_items[-1]

    middle_index = len(valid_items) // 2

    # Prefer a low-cloud observation around the
    # temporal middle of the sequence.
    candidates = valid_items[
        max(1, middle_index - 1):
        min(len(valid_items) - 1, middle_index + 2)
    ]

    middle = min(
        candidates,
        key=lambda item: (
            item.get("cloud_cover")
            if item.get("cloud_cover") is not None
            else 100.0
        ),
    )

    return [
        first["date"],
        middle["date"],
        last["date"],
    ]


class EOAgent:
    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
        max_steps: int = 8,
    ):
        self.model = model or os.getenv(
            "OLLAMA_MODEL",
            "qwen3:4b",
        )

        self.host = host or os.getenv(
            "OLLAMA_HOST",
            "http://localhost:11434",
        )

        self.max_steps = max_steps
        self.client = Client(host=self.host)

    def run(self, question: str) -> dict:

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": question,
            },
        ]

        trace = []
        artifacts = {}

        q = question.lower()

        analysis_requested = any(
            term in q
            for term in [
                "prithvi",
                "geofm",
                "temporal change",
                "representation change",
                "change detection",
                "analyze",
                "analyse",
            ]
        )

        analysis_completed = False

        last_search_result = None
        last_search_args = None

        for step in range(self.max_steps):

            response = self.client.chat(
                model=self.model,
                messages=messages,
                tools=TOOL_SCHEMAS,
            )

            message = response.message
            messages.append(message)

            tool_calls = (
                getattr(message, "tool_calls", None)
                or []
            )

            # -------------------------------------------------
            # Qwen is attempting to produce final answer
            # -------------------------------------------------

            if not tool_calls:

                # If Prithvi was requested and search has already
                # succeeded, deterministically complete analysis.
                if (
                    analysis_requested
                    and not analysis_completed
                    and last_search_result is not None
                ):

                    items = last_search_result.get(
                        "items",
                        [],
                    )

                    try:
                        selected_dates = (
                            _select_three_observations(items)
                        )
                    except Exception as exc:
                        return {
                            "answer": (
                                "Unable to select three suitable "
                                "Sentinel-2 observations for "
                                "Prithvi analysis."
                            ),
                            "trace": trace,
                            "model": self.model,
                            "artifacts": artifacts,
                            "error": str(exc),
                        }

                    analyze_fn = TOOL_FUNCTIONS.get(
                        "analyze_temporal_aoi"
                    )

                    if analyze_fn is None:
                        return {
                            "answer": (
                                "The analyze_temporal_aoi tool "
                                "is not registered."
                            ),
                            "trace": trace,
                            "model": self.model,
                            "artifacts": artifacts,
                        }

                    bbox = last_search_args.get("bbox")

                    analyze_args = {
                        "bbox": bbox,
                        "dates": selected_dates,
                    }

                    try:
                        result = analyze_fn(
                            **analyze_args
                        )

                    except Exception as exc:
                        result = {
                            "error":
                                type(exc).__name__,
                            "message":
                                str(exc),
                        }

                    if (
                        isinstance(result, dict)
                        and "error" not in result
                    ):
                        analysis_completed = True

                        tool_artifacts = result.get(
                            "artifacts",
                            {},
                        )

                        if isinstance(
                            tool_artifacts,
                            dict,
                        ):
                            artifacts.update(
                                tool_artifacts
                            )

                        result_summary = {
                            "dates":
                                result.get("dates"),

                            "model":
                                result.get(
                                    "geofm",
                                    {},
                                ).get(
                                    "model"
                                ),

                            "input_shape":
                                result.get(
                                    "geofm",
                                    {},
                                ).get(
                                    "input_shape"
                                ),

                            "start_end_cosine_distance":
                                result.get(
                                    "geofm",
                                    {},
                                ).get(
                                    "summary",
                                    {},
                                ).get(
                                    "start_end_cosine_distance"
                                ),

                            "prithvi_change_geotiff":
                                artifacts.get(
                                    "prithvi_change_geotiff"
                                ),
                        }

                    else:
                        result_summary = result

                    trace.append(
                        {
                            "step": step + 1,
                            "tool": "analyze_temporal_aoi",
                            "arguments": analyze_args,
                            "result_summary":
                                result_summary,
                            "selection_strategy":
                                "deterministic_temporal_spread",
                        }
                    )

                    messages.append(
                        {
                            "role": "tool",
                            "tool_name":
                                "analyze_temporal_aoi",
                            "content": json.dumps(
                                result,
                                default=str,
                            ),
                        }
                    )

                    # Let Qwen interpret the real
                    # deterministic Prithvi evidence.
                    continue

                # Safe final answer
                return {
                    "answer": message.content,
                    "trace": trace,
                    "model": self.model,
                    "artifacts": artifacts,
                }

            # -------------------------------------------------
            # Execute Qwen-requested tools
            # -------------------------------------------------

            for call in tool_calls:

                name = call.function.name

                args = (
                    call.function.arguments
                    or {}
                )

                if isinstance(args, str):
                    args = json.loads(args)

                fn = TOOL_FUNCTIONS.get(name)

                if fn is None:
                    result = {
                        "error":
                            f"Unknown tool: {name}"
                    }

                else:
                    try:
                        result = fn(**args)

                    except Exception as exc:
                        result = {
                            "error":
                                type(exc).__name__,
                            "message":
                                str(exc),
                        }

                # ---------------------------------------------
                # Store successful search
                # ---------------------------------------------

                if (
                    name == "search_sentinel2"
                    and isinstance(result, dict)
                    and "error" not in result
                ):

                    last_search_result = result
                    last_search_args = args

                    result_summary = {
                        "count":
                            result.get("count"),

                        "dates": [
                            item.get("date")
                            for item
                            in result.get(
                                "items",
                                [],
                            )
                        ],
                    }

                # ---------------------------------------------
                # Store successful analysis
                # ---------------------------------------------

                elif (
                    name == "analyze_temporal_aoi"
                    and isinstance(result, dict)
                    and "error" not in result
                ):

                    analysis_completed = True

                    tool_artifacts = result.get(
                        "artifacts",
                        {},
                    )

                    if isinstance(
                        tool_artifacts,
                        dict,
                    ):
                        artifacts.update(
                            tool_artifacts
                        )

                    result_summary = {
                        "dates":
                            result.get("dates"),

                        "model":
                            result.get(
                                "geofm",
                                {},
                            ).get(
                                "model"
                            ),

                        "input_shape":
                            result.get(
                                "geofm",
                                {},
                            ).get(
                                "input_shape"
                            ),

                        "start_end_cosine_distance":
                            result.get(
                                "geofm",
                                {},
                            ).get(
                                "summary",
                                {},
                            ).get(
                                "start_end_cosine_distance"
                            ),

                        "prithvi_change_geotiff":
                            artifacts.get(
                                "prithvi_change_geotiff"
                            ),
                    }

                elif (
                    isinstance(result, dict)
                    and "error" in result
                ):
                    result_summary = result

                else:
                    result_summary = "success"

                trace.append(
                    {
                        "step": step + 1,
                        "tool": name,
                        "arguments": args,
                        "result_summary":
                            result_summary,
                    }
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_name": name,
                        "content": json.dumps(
                            result,
                            default=str,
                        ),
                    }
                )

        return {
            "answer": (
                "Maximum tool steps reached before "
                "the requested analysis completed."
            ),
            "trace": trace,
            "model": self.model,
            "artifacts": artifacts,
        }

#"""Minimal Ollama/Qwen tool-calling loop."""
#
#from __future__ import annotations
#
#import json
#import os
#
#from ollama import Client
#
#from agent.prompts import SYSTEM_PROMPT
#from agent.tool_registry import TOOL_FUNCTIONS, TOOL_SCHEMAS
#
#
#class EOAgent:
#    def __init__(self, model: str | None = None, host: str | None = None, max_steps: int = 6):
#        self.model = model or os.getenv("OLLAMA_MODEL", "qwen3:4b")
#        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
#        self.max_steps = max_steps
#        self.client = Client(host=self.host)
#
#    def run(self, question: str) -> dict:
#        messages = [
#            {"role": "system", "content": SYSTEM_PROMPT},
#            {"role": "user", "content": question},
#        ]
#        trace = []
#
#        for step in range(self.max_steps):
#            response = self.client.chat(
#                model=self.model,
#                messages=messages,
#                tools=TOOL_SCHEMAS,
#            )
#            message = response.message
#            messages.append(message)
#
#            tool_calls = getattr(message, "tool_calls", None) or []
#            if not tool_calls:
#                return {"answer": message.content, "trace": trace, "model": self.model}
#
#            for call in tool_calls:
#                name = call.function.name
#                args = call.function.arguments or {}
#                if isinstance(args, str):
#                    args = json.loads(args)
#
#                fn = TOOL_FUNCTIONS.get(name)
#                if fn is None:
#                    result = {"error": f"Unknown tool: {name}"}
#                else:
#                    try:
#                        result = fn(**args)
#                    except Exception as exc:
#                        result = {"error": type(exc).__name__, "message": str(exc)}
#
#                trace.append({
#                    "step": step + 1,
#                    "tool": name,
#                    "arguments": args,
#                    "result_summary": result if isinstance(result, dict) and "error" in result else "success",
#                })
#                messages.append({
#                    "role": "tool",
#                    "tool_name": name,
#                    "content": json.dumps(result, default=str),
#                })
#
#        return {
#            "answer": "Maximum tool steps reached before a final response.",
#            "trace": trace,
#            "model": self.model,
#        }
