"""Minimal Ollama/Qwen tool-calling loop."""

from __future__ import annotations

import json
import os

from ollama import Client

from agent.prompts import SYSTEM_PROMPT
from agent.tool_registry import TOOL_FUNCTIONS, TOOL_SCHEMAS


class EOAgent:
    def __init__(self, model: str | None = None, host: str | None = None, max_steps: int = 6):
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen3")
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.max_steps = max_steps
        self.client = Client(host=self.host)

    def run(self, question: str) -> dict:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        trace = []

        for step in range(self.max_steps):
            response = self.client.chat(
                model=self.model,
                messages=messages,
                tools=TOOL_SCHEMAS,
            )
            message = response.message
            messages.append(message)

            tool_calls = getattr(message, "tool_calls", None) or []
            if not tool_calls:
                return {"answer": message.content, "trace": trace, "model": self.model}

            for call in tool_calls:
                name = call.function.name
                args = call.function.arguments or {}
                if isinstance(args, str):
                    args = json.loads(args)

                fn = TOOL_FUNCTIONS.get(name)
                if fn is None:
                    result = {"error": f"Unknown tool: {name}"}
                else:
                    try:
                        result = fn(**args)
                    except Exception as exc:
                        result = {"error": type(exc).__name__, "message": str(exc)}

                trace.append({
                    "step": step + 1,
                    "tool": name,
                    "arguments": args,
                    "result_summary": result if isinstance(result, dict) and "error" in result else "success",
                })
                messages.append({
                    "role": "tool",
                    "tool_name": name,
                    "content": json.dumps(result, default=str),
                })

        return {
            "answer": "Maximum tool steps reached before a final response.",
            "trace": trace,
            "model": self.model,
        }
