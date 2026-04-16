param(
    [string]$Model = "qwen3:8b",
    [string]$ModelStore = "E:\ollama\models",
    [switch]$SkipPull,
    [switch]$NoLaunch
)

$ErrorActionPreference = "Stop"

function Find-OllamaExe {
    $candidates = @(
        "C:\Users\$env:USERNAME\AppData\Local\Programs\Ollama\ollama.exe",
        "C:\Program Files\Ollama\ollama.exe"
    )

    foreach ($p in $candidates) {
        if (Test-Path $p) { return $p }
    }

    $cmd = Get-Command ollama -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }

    throw "Ollama not found. Install it first."
}

function Find-ClaudeExe {
    $candidates = @(
        "C:\Users\$env:USERNAME\AppData\Local\Microsoft\WinGet\Packages\Anthropic.ClaudeCode_Microsoft.Winget.Source_8wekyb3d8bbwe\claude.exe",
        "C:\Users\$env:USERNAME\AppData\Local\Programs\Claude\claude.exe"
    )

    foreach ($p in $candidates) {
        if (Test-Path $p) { return $p }
    }

    $cmd = Get-Command claude -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }

    throw "Claude Code CLI not found. Reinstall with winget install Anthropic.ClaudeCode"
}

function Ensure-OllamaApi {
    try {
        Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:11434/api/tags" -TimeoutSec 3 | Out-Null
        return
    }
    catch {
        Start-Process -FilePath $script:OllamaExe -ArgumentList "serve" -WindowStyle Hidden | Out-Null
        Start-Sleep -Seconds 3
        Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:11434/api/tags" -TimeoutSec 5 | Out-Null
    }
}

$script:OllamaExe = Find-OllamaExe
$claudeExe = Find-ClaudeExe

New-Item -ItemType Directory -Path $ModelStore -Force | Out-Null
[Environment]::SetEnvironmentVariable("OLLAMA_MODELS", $ModelStore, "User")
$env:OLLAMA_MODELS = $ModelStore

Ensure-OllamaApi

if (-not $SkipPull) {
    $installed = & $script:OllamaExe list
    if ($installed -notmatch [regex]::Escape($Model)) {
        & $script:OllamaExe pull $Model
    }
}

$probePayload = @{ model = $Model; prompt = "Reply with exactly: local workflow ready"; stream = $false } | ConvertTo-Json
$probeResult = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/generate" -Method Post -ContentType "application/json" -Body $probePayload

Write-Host "Ollama model reply: $($probeResult.response)"
Write-Host "Launching Claude Code with workspace MCP config..."
Write-Host "Tip: use qwen3:8b for local execution; qwen3:32b may require more RAM/VRAM."
Write-Host "Tip: in Claude, ask it to call ask_local_qwen for batch/repetitive tasks."

if ($NoLaunch) {
    Write-Host "NoLaunch enabled. Environment checks passed."
    exit 0
}

& $claudeExe
