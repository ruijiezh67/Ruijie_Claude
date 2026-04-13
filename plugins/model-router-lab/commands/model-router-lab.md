---
allowed-tools: Bash(python:*), Bash(git diff:*), Bash(git ls-files:*)
description: Run Model Router Lab end-to-end with SLA-aware routing
argument-hint: Optional model list, e.g. qwen3:8b,qwen3:32b
---

## Context

- Workspace root: !`pwd`
- Optional models: $ARGUMENTS

## Your task

Run Model Router Lab with repository sampling, benchmark, policy build, and report rendering.

1. Generate dataset from current repo:
- `python projects/model-router-lab/routerlab.py sample-tasks --max-files 20 --out projects/model-router-lab/datasets/repo_tasks.jsonl`

2. Choose model list:
- Use `$ARGUMENTS` if provided
- Otherwise use `qwen3:8b`

3. Run benchmark:
- `python projects/model-router-lab/routerlab.py benchmark --dataset projects/model-router-lab/datasets/repo_tasks.jsonl --models <models> --judge-model qwen3:8b --min-score 7 --det-weight 0.4 --sla-max-latency 45 --prices qwen3:8b=0.10,qwen3:32b=0.45 --out projects/model-router-lab/outputs/benchmark.json`

4. Build policy:
- `python projects/model-router-lab/routerlab.py build-policy --report projects/model-router-lab/outputs/benchmark.json --min-score 7 --sla-max-latency 45 --out projects/model-router-lab/outputs/policy.json`

5. Render report:
- `python projects/model-router-lab/routerlab.py render-report --report projects/model-router-lab/outputs/benchmark.json --out projects/model-router-lab/outputs/report.html`

6. Print a concise summary:
- Default model
- Escalation model
- Pass rate and P95 latency by model
- Report file path
