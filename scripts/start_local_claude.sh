#!/usr/bin/env bash
set -euo pipefail

MODEL="qwen3:8b"
MODEL_STORE_RAW="${OLLAMA_MODELS:-E:/ollama/models}"
SKIP_PULL=0
NO_LAUNCH=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="${2:-}"
      shift 2
      ;;
    --model-store)
      MODEL_STORE_RAW="${2:-}"
      shift 2
      ;;
    --skip-pull)
      SKIP_PULL=1
      shift
      ;;
    --no-launch)
      NO_LAUNCH=1
      shift
      ;;
    -h|--help)
      echo "Usage: bash scripts/start_local_claude.sh [--model qwen3:8b] [--model-store E:/ollama/models] [--skip-pull] [--no-launch]"
      exit 0
      ;;
    *)
      # Backward-compatible positional model argument.
      MODEL="$1"
      shift
      ;;
  esac
done

# Common typo guard: user may type qwen3:8b~
MODEL="${MODEL%~}"

to_windows_path() {
  local p="$1"
  if [[ "$p" =~ ^[A-Za-z]:[\\/] ]]; then
    echo "${p//\//\\}"
    return
  fi
  if command -v cygpath >/dev/null 2>&1; then
    cygpath -w "$p"
    return
  fi
  echo "$p"
}

find_ollama() {
  if command -v ollama >/dev/null 2>&1; then
    command -v ollama
    return
  fi
  local candidates=(
    "/c/Users/$USERNAME/AppData/Local/Programs/Ollama/ollama.exe"
    "/c/Program Files/Ollama/ollama.exe"
  )
  for p in "${candidates[@]}"; do
    if [[ -x "$p" || -f "$p" ]]; then
      echo "$p"
      return
    fi
  done
  return 1
}

find_claude() {
  if command -v claude >/dev/null 2>&1; then
    command -v claude
    return
  fi
  local candidates=(
    "/c/Users/$USERNAME/AppData/Local/Microsoft/WinGet/Packages/Anthropic.ClaudeCode_Microsoft.Winget.Source_8wekyb3d8bbwe/claude.exe"
    "/c/Users/$USERNAME/AppData/Local/Programs/Claude/claude.exe"
  )
  for p in "${candidates[@]}"; do
    if [[ -x "$p" || -f "$p" ]]; then
      echo "$p"
      return
    fi
  done
  return 1
}

OLLAMA_BIN="$(find_ollama || true)"
if [[ -z "$OLLAMA_BIN" ]]; then
  echo "Ollama is not installed or not in PATH"
  exit 1
fi

CLAUDE_BIN="$(find_claude || true)"
if [[ -z "$CLAUDE_BIN" ]]; then
  echo "Claude CLI not found in PATH or default install path"
  exit 1
fi

MODEL_STORE_WIN="$(to_windows_path "$MODEL_STORE_RAW")"
export OLLAMA_MODELS="$MODEL_STORE_WIN"

if command -v cygpath >/dev/null 2>&1; then
  mkdir -p "$(cygpath -u "$MODEL_STORE_WIN")"
fi

if ! curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  nohup "$OLLAMA_BIN" serve >/tmp/ollama-serve.log 2>&1 &
  sleep 3
fi

if [[ "$SKIP_PULL" -eq 0 ]]; then
  if ! "$OLLAMA_BIN" list | grep -q "$MODEL"; then
    "$OLLAMA_BIN" pull "$MODEL"
  fi
fi

PROBE=$(curl -sS http://127.0.0.1:11434/api/generate \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"prompt\":\"Reply with exactly: local workflow ready\",\"stream\":false}" \
  | sed -n 's/.*"response":"\([^"]*\)".*/\1/p')

echo "Ollama model reply: ${PROBE:-ok}"

if [[ "$NO_LAUNCH" -eq 1 ]]; then
  echo "NoLaunch enabled. Environment checks passed."
  exit 0
fi

echo "Launching Claude Code..."
"$CLAUDE_BIN"
