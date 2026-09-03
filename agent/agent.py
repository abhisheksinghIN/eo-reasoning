"""Minimal Ollama/Qwen tool-calling loop."""

from __future__ import annotations

import json
import os

from ollama import Client

from agent.prompts import SYSTEM_PROMPT
from agent.tool_registry import (
    TOOL_FUNCTIONS,
    TOOL_SCHEMAS,
)


class EOAgent:
    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
        max_steps: int = 6,
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

        # -------------------------------------------------
        # Determine whether the user actually requested
        # GeoFM / Prithvi analysis.
        # -------------------------------------------------

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

        # Store dates returned by Sentinel-2 search.
        searched_dates: set[str] = set()

        for step in range(self.max_steps):

            response = self.client.chat(
                model=self.model,
                messages=messages,
                tools=TOOL_SCHEMAS,
            )

            message = response.message
            messages.append(message)

            tool_calls = (
                getattr(
                    message,
                    "tool_calls",
                    None,
                )
                or []
            )

            # =================================================
            # Qwen wants to produce its final answer
            # =================================================

            if not tool_calls:

                # ---------------------------------------------
                # Prevent premature completion.
                # ---------------------------------------------

                if (
                    analysis_requested
                    and not analysis_completed
                ):

                    if searched_dates:

                        if len(searched_dates) < 3:
                            return {
                                "answer": (
                                    "Prithvi temporal analysis requires "
                                    "three Sentinel-2 observations, but "
                                    f"only {len(searched_dates)} suitable "
                                    "acquisition date(s) were found."
                                ),
                                "trace": trace,
                                "model": self.model,
                                "artifacts": artifacts,
                            }

                        messages.append(
                            {
                                "role": "system",
                                "content": (
                                    "The requested GeoFM workflow is "
                                    "not complete. You have searched "
                                    "Sentinel-2 observations but have "
                                    "not run Prithvi analysis. "
                                    "Select exactly three acquisition "
                                    "dates ONLY from the Sentinel-2 "
                                    "dates already returned by the tool. "
                                    "Prefer good temporal coverage. "
                                    "Then call analyze_temporal_aoi. "
                                    "Do not provide a final numerical "
                                    "answer until that tool succeeds."
                                ),
                            }
                        )

                    else:

                        messages.append(
                            {
                                "role": "system",
                                "content": (
                                    "The user requested Prithvi/GeoFM "
                                    "analysis, but no deterministic "
                                    "analysis has been executed. "
                                    "Use the available EO tools. "
                                    "If suitable acquisition dates are "
                                    "not known, call search_sentinel2 "
                                    "first. Then call "
                                    "analyze_temporal_aoi with exactly "
                                    "three actual acquisition dates. "
                                    "Do not invent numerical results."
                                ),
                            }
                        )

                    continue

                # ---------------------------------------------
                # Safe final response
                # ---------------------------------------------

                return {
                    "answer": message.content,
                    "trace": trace,
                    "model": self.model,
                    "artifacts": artifacts,
                }

            # =================================================
            # Execute Qwen tool calls
            # =================================================

            for call in tool_calls:

                name = call.function.name

                args = (
                    call.function.arguments
                    or {}
                )

                if isinstance(args, str):
                    args = json.loads(args)

                fn = TOOL_FUNCTIONS.get(name)

                # ---------------------------------------------
                # Validate analyze_temporal_aoi dates
                # ---------------------------------------------

                if (
                    name == "analyze_temporal_aoi"
                    and searched_dates
                ):

                    requested_dates = args.get(
                        "dates",
                        [],
                    )

                    if len(requested_dates) != 3:

                        result = {
                            "error": "InvalidTemporalSelection",
                            "message": (
                                "Prithvi temporal analysis requires "
                                "exactly three acquisition dates."
                            ),
                        }

                    elif not all(
                        date in searched_dates
                        for date in requested_dates
                    ):

                        invalid_dates = [
                            date
                            for date in requested_dates
                            if date not in searched_dates
                        ]

                        result = {
                            "error": "InvalidAcquisitionDate",
                            "message": (
                                "The analysis attempted to use dates "
                                "that were not returned by the "
                                "Sentinel-2 search: "
                                f"{invalid_dates}. "
                                "Use only actual searched acquisition "
                                "dates."
                            ),
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

                elif fn is None:

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

                # =================================================
                # Process successful search results
                # =================================================

                if (
                    name == "search_sentinel2"
                    and isinstance(result, dict)
                    and "error" not in result
                ):

                    searched_dates = {
                        item.get("date")
                        for item in result.get(
                            "items",
                            [],
                        )
                        if item.get("date")
                    }

                # =================================================
                # Process successful analysis
                # =================================================

                if (
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

                # =================================================
                # Build useful trace
                # =================================================

                if (
                    name == "search_sentinel2"
                    and isinstance(result, dict)
                    and "error" not in result
                ):

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

                elif (
                    name == "analyze_temporal_aoi"
                    and isinstance(result, dict)
                    and "error" not in result
                ):

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

                        "artifacts": {
                            "prithvi_change_geotiff":
                                artifacts.get(
                                    "prithvi_change_geotiff"
                                ),

                            "ndvi_change_geotiff":
                                artifacts.get(
                                    "ndvi_change_geotiff"
                                ),
                        },
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

                # =================================================
                # Return deterministic tool evidence to Qwen
                # =================================================

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

        # =====================================================
        # Maximum number of calls reached
        # =====================================================

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
