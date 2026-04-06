#!/usr/bin/env python3
import json
import urllib.error
import urllib.request

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("local-qwen-worker")


def _call_ollama(prompt: str, model: str, temperature: float) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
        },
    }

    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(
            "Cannot reach Ollama at http://127.0.0.1:11434. Please start Ollama first."
        ) from e

    text = data.get("response", "")
    if not text:
        raise RuntimeError("Ollama returned an empty response.")
    return text


@mcp.tool()
def ask_local_qwen(
    prompt: str,
    model: str = "qwen2.5:7b",
    temperature: float = 0.2,
) -> str:
    """Send a prompt to local Ollama model (Qwen by default) and return the generated text."""
    return _call_ollama(prompt=prompt, model=model, temperature=temperature)


if __name__ == "__main__":
    mcp.run(transport="stdio")
