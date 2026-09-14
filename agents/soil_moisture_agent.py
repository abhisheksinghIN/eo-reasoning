"""Local Mistral agent for EO soil-moisture orchestration."""

from __future__ import annotations

import argparse
import json
import os

from ollama import chat

from tools.soil_moisture_tool import (
    analyze_soil_moisture_aoi,
)


DEFAULT_MODEL = os.getenv(
    "EO_AGENT_MODEL",
    "ministral-3:8b",
)


SYSTEM_PROMPT = """
You are EO-Reasoning, an Earth-observation orchestration assistant.

You do NOT estimate soil moisture yourself.

Your role is to:
1. understand the user's Earth-observation request;
2. call deterministic EO analysis tools with the exact requested
   spatial and temporal parameters;
3. interpret only the evidence returned by those tools;
4. clearly communicate uncertainty, provenance, and limitations.

SCIENTIFIC RULES

1. Never invent:
   - EO observations,
   - Sentinel acquisition dates,
   - satellite platforms,
   - orbit numbers,
   - backscatter values,
   - soil-moisture values,
   - GeoFM outputs,
   - JEPA outputs,
   - physics residuals.

2. Any quantitative Earth-observation value must come from a tool.

3. When a user requests quantitative soil-moisture analysis,
   call analyze_soil_moisture_aoi.

4. Preserve the user's bbox and dates exactly unless the tool
   itself reports that they cannot be used.

5. Distinguish clearly between:

   A. Observations
      Sentinel-1 measurements and acquisition metadata.

   B. GeoFM retrieval
      Soil-moisture output derived from the target-date
      TerraMind representation.

   C. JEPA prediction
      Soil-moisture output derived from the temporally predicted
      target representation.

   D. Physics consistency
      Agreement or disagreement between the predicted soil-moisture
      state and the SAR observation proxy.

   E. CLMS reference
      Copernicus Land Monitoring Service Surface Soil Moisture.

6. CLMS Surface Soil Moisture is a reference / weak-supervision
   product. Do not call it independent ground truth.

7. Never interpret latent cosine distance itself as:
   - soil moisture,
   - drought,
   - crop stress,
   - irrigation,
   - vegetation stress.

8. Never infer causes such as drought, irrigation, rainfall deficit,
   crop stress, flooding, or management practices unless an
   appropriate deterministic tool provides evidence for that claim.

9. MODEL STATUS RULE:

   If tool output contains:

       model.status == "untrained_prototype"

   then you MUST explicitly say that the retrieval and predictive
   soil-moisture numbers are developmental prototype outputs and
   are not scientifically validated estimates.

   In this case:
   - do not rank the model as accurate or inaccurate from one sample;
   - do not claim successful soil-moisture retrieval;
   - do not recommend operational use;
   - treat CLMS only as a contextual reference.

10. If:

       interpretation_status == "not_validated"

    explicitly state that the model requires training and
    held-out validation before scientific interpretation.

11. If a tool reports insufficient observations, explain that
    result rather than inventing missing acquisitions.

12. Do not strengthen a physics-proxy result into a physical
    causal conclusion.

13. Sentinel-2 is optional. Do not claim Sentinel-2 was used
    when the tool says sentinel2_used == false.

RESPONSE FORMAT

When analysis succeeds, organize the response as:

Scope
Observations
Model evidence
CLMS reference
Physics / temporal consistency
Interpretation and limitations

Keep observations and interpretations separate.

The interface is AI-assisted EO interpretation. Make it clear that
the narrative is model-generated from deterministic EO evidence.
"""


AVAILABLE_FUNCTIONS = {
    "analyze_soil_moisture_aoi":
        analyze_soil_moisture_aoi,
}


def run_agent(
    user_prompt: str,
    model_name: str = DEFAULT_MODEL,
    max_steps: int = 4,
    verbose: bool = True,
) -> str:
    """
    Run the local Mistral tool-calling agent.
    """

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    tools = list(
        AVAILABLE_FUNCTIONS.values()
    )

    for step in range(
        1,
        max_steps + 1,
    ):

        response = chat(
            model=model_name,
            messages=messages,
            tools=tools,
            options={
                "temperature": 0.1,
            },
        )

        # Preserve assistant message, including tool calls.
        messages.append(
            response.message
        )

        tool_calls = (
            response.message.tool_calls
            or []
        )

        # -------------------------------------------------
        # No further tool call -> final natural-language answer.
        # -------------------------------------------------

        if not tool_calls:

            content = (
                response.message.content
                or ""
            )

            return content.strip()

        # -------------------------------------------------
        # Execute requested deterministic tools.
        # -------------------------------------------------

        for tool_call in tool_calls:

            function_name = (
                tool_call.function.name
            )

            arguments = (
                tool_call.function.arguments
                or {}
            )

            if verbose:
                print(
                    "\n"
                    f"[agent step {step}] "
                    f"tool={function_name}"
                )

                print(
                    "[arguments]"
                )

                print(
                    json.dumps(
                        arguments,
                        indent=2,
                    )
                )

            function = (
                AVAILABLE_FUNCTIONS.get(
                    function_name
                )
            )

            if function is None:

                result = {
                    "status": "error",
                    "message": (
                        f"Unknown tool: "
                        f"{function_name}"
                    ),
                }

            else:

                try:

                    result = function(
                        **arguments
                    )

                except Exception as exc:

                    result = {
                        "status": "error",
                        "tool": (
                            function_name
                        ),
                        "error_type": (
                            type(
                                exc
                            ).__name__
                        ),
                        "message": str(
                            exc
                        ),
                    }

            if verbose:
                print(
                    "[tool result status]",
                    result.get(
                        "status",
                        "unknown",
                    ),
                )

            # Return deterministic evidence to Mistral.
            messages.append(
                {
                    "role": "tool",
                    "tool_name": (
                        function_name
                    ),
                    "content": json.dumps(
                        result,
                        default=str,
                    ),
                }
            )

    raise RuntimeError(
        "Agent exceeded maximum tool-calling steps."
    )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "EO-Reasoning local Mistral "
            "soil-moisture agent"
        )
    )

    parser.add_argument(
        "prompt",
        nargs="*",
        help=(
            "Natural-language EO request. "
            "If omitted, interactive input is used."
        ),
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "Ollama model name. "
            f"Default: {DEFAULT_MODEL}"
        ),
    )

    parser.add_argument(
        "--quiet-tools",
        action="store_true",
        help=(
            "Hide tool trace."
        ),
    )

    args = parser.parse_args()

    if args.prompt:

        prompt = " ".join(
            args.prompt
        )

    else:

        prompt = input(
            "EO-Reasoning> "
        ).strip()

    if not prompt:
        raise SystemExit(
            "Prompt cannot be empty."
        )

    answer = run_agent(
        user_prompt=prompt,
        model_name=args.model,
        verbose=not args.quiet_tools,
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "EO-Reasoning"
    )

    print(
        "=" * 72
    )

    print(
        answer
    )


if __name__ == "__main__":
    main()