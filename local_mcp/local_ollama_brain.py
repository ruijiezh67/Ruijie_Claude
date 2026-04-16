#!/usr/bin/env python3
import argparse
import json
import sys
import urllib.error
import urllib.request


def call_ollama(prompt: str, model: str, temperature: float) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }

    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("response", "")


def ensure_ollama_up() -> None:
    req = urllib.request.Request("http://127.0.0.1:11434/api/tags", method="GET")
    try:
        urllib.request.urlopen(req, timeout=4)
    except urllib.error.URLError as e:
        raise RuntimeError(
            "Ollama service is not reachable at http://127.0.0.1:11434"
        ) from e


def repl(model: str, temperature: float, system_prompt: str) -> int:
    print("Local Brain (Ollama) ready.")
    print("Commands: /exit, /model <name>, /temp <value>, /help")
    print(f"Current model: {model}")

    current_model = model
    current_temp = temperature

    while True:
        try:
            user = input("you> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nbye")
            return 0

        if not user:
            continue

        if user in {"/exit", "/quit"}:
            print("bye")
            return 0

        if user.startswith("/model "):
            current_model = user.split(" ", 1)[1].strip()
            print(f"model set to: {current_model}")
            continue

        if user.startswith("/temp "):
            value = user.split(" ", 1)[1].strip()
            try:
                current_temp = float(value)
                print(f"temperature set to: {current_temp}")
            except ValueError:
                print("invalid temperature")
            continue

        if user == "/help":
            print("/exit, /quit, /model <name>, /temp <float>")
            continue

        prompt = f"System: {system_prompt}\n\nUser: {user}\nAssistant:"
        try:
            answer = call_ollama(prompt, current_model, current_temp)
        except Exception as e:  # pragma: no cover
            print(f"error: {e}")
            continue

        print("local> " + (answer.strip() or "[empty response]"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pure local Ollama REPL")
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--system",
        default=(
            "You are a local coding assistant. Be concise, safe, and practical. "
            "When uncertain, say what to verify."
        ),
    )
    args = parser.parse_args()

    try:
        ensure_ollama_up()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    sys.exit(repl(args.model, args.temperature, args.system))
