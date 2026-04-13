#!/usr/bin/env python3
import argparse
import ast
import json
import re
import subprocess
import time
import urllib.error
import urllib.request
from collections import defaultdict
from html import escape
from pathlib import Path

OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_PRICES = {
    "qwen3:8b": 0.10,
    "qwen3:32b": 0.45,
}


def ollama_generate(
    model: str,
    prompt: str,
    temperature: float = 0.1,
    num_predict: int = 320,
) -> tuple[str, float]:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": num_predict},
    }
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.time()
    with urllib.request.urlopen(req, timeout=900) as resp:
        out = json.loads(resp.read().decode("utf-8"))
    latency = time.time() - start
    text = (out.get("response", "") or "").strip()
    if not text:
        # Some reasoning models may return content in `thinking` with an empty `response`.
        text = (out.get("thinking", "") or "").strip()
    return text, latency


def ensure_ollama() -> None:
    req = urllib.request.Request(f"{OLLAMA_URL}/api/tags", method="GET")
    try:
        urllib.request.urlopen(req, timeout=5)
    except urllib.error.URLError as exc:
        raise RuntimeError("Ollama service is not reachable on 127.0.0.1:11434") from exc


def ollama_models() -> list[str]:
    req = urllib.request.Request(f"{OLLAMA_URL}/api/tags", method="GET")
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return [m.get("name", "") for m in data.get("models", []) if m.get("name")]


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def estimate_cost(tokens: int, price_per_1m: float) -> float:
    return (tokens / 1_000_000.0) * price_per_1m


def write_json(path: str, obj: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")


def write_text(path: str, content: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(content, encoding="utf-8")


def run_cmd(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = int(round((pct / 100.0) * (len(s) - 1)))
    idx = max(0, min(len(s) - 1, idx))
    return s[idx]


def clip_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars]


def parse_json_fragment(raw: str) -> dict | None:
    raw = raw.strip()
    if not raw:
        return None
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except Exception:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        chunk = raw[start : end + 1]
        try:
            data = json.loads(chunk)
            return data if isinstance(data, dict) else None
        except Exception:
            return None
    return None


def model_priority(models: list[str], prices: dict[str, float]) -> list[str]:
    return sorted(models, key=lambda m: prices.get(m, prices.get("default", 1.0)))


def load_tasks(path: str) -> list[dict]:
    raw = Path(path).read_text(encoding="utf-8").strip()
    if not raw:
        return []
    if raw.startswith("["):
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    rows = []
    decoder = json.JSONDecoder()
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.endswith("\\n"):
            s = s[:-2].rstrip()
        pos = 0
        while pos < len(s):
            while pos < len(s) and s[pos].isspace():
                pos += 1
            if pos >= len(s):
                break
            obj, nxt = decoder.raw_decode(s, pos)
            rows.append(obj)
            pos = nxt
    return rows


def extract_code_block(text: str) -> str:
    m = re.search(r"```(?:python)?\n([\s\S]*?)```", text, re.IGNORECASE)
    return m.group(1).strip() if m else ""


def deterministic_score(task: dict, answer: str) -> tuple[int, str]:
    score = 10
    reasons = []
    ttype = task.get("type", "general")
    expected = str(task.get("expected", "")).lower()
    answer_lower = answer.lower()

    if expected:
        keywords = [k.strip() for k in re.split(r"[;,]", expected) if k.strip()]
        miss = 0
        for kw in keywords[:4]:
            parts = [p for p in re.split(r"\s+", kw) if len(p) > 3]
            if parts and not any(p in answer_lower for p in parts):
                miss += 1
        if miss:
            if ttype.startswith("coding"):
                score -= min(4, miss * 2)
            else:
                score -= min(2, (miss + 1) // 2)
            reasons.append(f"missing expected keywords: {miss}")

    if ttype.startswith("coding"):
        code = extract_code_block(answer)
        if not code:
            score -= 3
            reasons.append("missing code block")
        else:
            try:
                ast.parse(code)
                score += 1
                reasons.append("valid python code block")
            except Exception:
                score -= 3
                reasons.append("code block failed python parse")
    else:
        if len(answer.strip()) >= 500:
            score += 1
            reasons.append("long-form coverage bonus")

    min_len = 80 if ttype.startswith("coding") else 40
    if len(answer.strip()) < min_len:
        score -= 1
        reasons.append("answer too short")

    final = max(1, min(10, score))
    return final, "; ".join(reasons) if reasons else "deterministic checks passed"


def judge_answer(
    judge_model: str,
    task: dict,
    answer: str,
    judge_max_tokens: int,
    prompt_max_chars: int,
) -> tuple[int, str]:
    rubric = task.get("rubric", "Return score from 1 to 10.")
    short_task = clip_text(task.get("prompt", ""), prompt_max_chars)
    short_answer = clip_text(answer, prompt_max_chars)
    prompt = (
        "Score candidate answer from 1 to 10.\n"
        "Return EXACTLY this two-line format:\n"
        "SCORE:<integer 1-10>\n"
        "REASON:<short reason>\n"
        f"Task: {short_task}\n"
        f"Expected (optional): {task.get('expected', '')}\n"
        f"Rubric: {rubric}\n"
        f"Candidate answer: {short_answer}\n"
        "Output now."
    )
    raw, _ = ollama_generate(judge_model, prompt, temperature=0.0, num_predict=judge_max_tokens)
    try:
        m = re.search(r"SCORE\s*:\s*(\d{1,2})", raw, re.IGNORECASE)
        if m:
            score = int(m.group(1))
        else:
            m2 = re.search(r"\b([1-9]|10)\b", raw)
            if not m2:
                raise ValueError("no-score")
            score = int(m2.group(1))
        rm = re.search(r"REASON\s*:\s*(.*)", raw, re.IGNORECASE)
        reason = rm.group(1).strip() if rm else raw.strip()[:160]
        return max(1, min(10, score)), reason
    except Exception:
        return 0, "Judge output parsing failed"


def sample_tasks_from_repo(args: argparse.Namespace) -> int:
    changed = run_cmd(["git", "diff", "--name-only", "HEAD"])
    files = [x.strip() for x in changed.splitlines() if x.strip()]
    if not files:
        tracked = run_cmd(["git", "ls-files"])
        files = [x.strip() for x in tracked.splitlines() if x.strip()]

    files = files[: args.max_files]
    tasks = []
    idx = 1

    for rel in files:
        p = Path(rel)
        if not p.exists() or p.is_dir():
            continue
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf", ".lock", ".pyc"}:
            continue

        snippet = p.read_text(encoding="utf-8", errors="ignore")[:2000]
        compact = snippet.replace("\r\n", "\n").replace("\n", "\\n")
        task_type = "coding_hard" if len(snippet) > 1000 else "coding_light"
        tasks.append(
            {
                "id": f"repo-{idx}",
                "type": task_type,
                "prompt": (
                    f"File: {rel}\\n"
                    "Propose one concrete improvement with patch-like detail.\\n"
                    f"SNIPPET:\\n{compact}"
                ),
                "expected": "Provide actionable improvement with rationale",
                "rubric": "Actionability, correctness, and technical depth.",
            }
        )
        idx += 1

    tasks.extend(
        [
            {
                "id": f"gen-{idx}",
                "type": "summarization",
                "prompt": "Summarize key technical risks in shipping a local-first AI coding product to teams.",
                "expected": "Mentions reliability, evaluation bias, support cost, compliance.",
                "rubric": "Coverage and practical insight.",
            },
            {
                "id": f"gen-{idx+1}",
                "type": "general",
                "prompt": "Give a 7-day launch plan to get first 100 GitHub stars for a devtool.",
                "expected": "Includes demo, docs, channels, feedback loop, release cadence.",
                "rubric": "Practicality and sequencing.",
            },
        ]
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(t, ensure_ascii=False) for t in tasks) + "\n", encoding="utf-8")
    print(f"Saved sampled dataset: {args.out}")
    print(f"Task count: {len(tasks)}")
    return 0


def parse_prices(price_text: str) -> dict:
    prices = dict(DEFAULT_PRICES)
    if not price_text:
        return prices
    for pair in price_text.split(","):
        if "=" not in pair:
            continue
        model, value = pair.split("=", 1)
        model = model.strip()
        try:
            prices[model] = float(value.strip())
        except ValueError:
            pass
    return prices


def run_benchmark(args: argparse.Namespace) -> int:
    ensure_ollama()
    tasks = load_tasks(args.dataset)
    if not tasks:
        print("Dataset is empty")
        return 2

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    prices = parse_prices(args.prices)
    ordered_models = model_priority(models, prices)
    baseline_model = ordered_models[0]
    escalation_model = ordered_models[-1]

    result = {
        "meta": {
            "dataset": args.dataset,
            "models": models,
            "judge_model": args.judge_model,
            "min_score": args.min_score,
            "sla_max_latency_sec": args.sla_max_latency,
            "deterministic_weight": args.det_weight,
            "prompt_max_chars": args.prompt_max_chars,
            "answer_max_tokens": args.answer_max_tokens,
            "judge_max_tokens": args.judge_max_tokens,
            "auto_escalate": args.auto_escalate,
            "prices_per_1m_tokens": prices,
            "timestamp": int(time.time()),
        },
        "tasks": [],
        "summary": {},
        "summary_by_type": {},
        "recommended_routing": {},
    }

    scores = defaultdict(list)
    lats = defaultdict(list)
    costs = defaultdict(float)
    by_type = defaultdict(lambda: defaultdict(list))
    by_type_lats = defaultdict(lambda: defaultdict(list))

    for task in tasks:
        ttype = task.get("type", "general")
        task_out = {"id": task.get("id"), "type": ttype, "runs": []}

        clipped_prompt = clip_text(task.get("prompt", ""), args.prompt_max_chars)
        for model in ordered_models:
            answer, latency = ollama_generate(
                model,
                clipped_prompt,
                temperature=args.temperature,
                num_predict=args.answer_max_tokens,
            )
            if not answer.strip():
                answer, retry_latency = ollama_generate(
                    model,
                    task.get("prompt", ""),
                    temperature=max(0.05, args.temperature),
                    num_predict=max(320, args.answer_max_tokens),
                )
                latency += retry_latency
            llm_score, llm_reason = judge_answer(
                args.judge_model,
                task,
                answer,
                judge_max_tokens=args.judge_max_tokens,
                prompt_max_chars=args.prompt_max_chars,
            )
            det_score, det_reason = deterministic_score(task, answer)
            if llm_score < 1:
                llm_score = det_score
            final_score = int(round(llm_score * (1 - args.det_weight) + det_score * args.det_weight))
            final_score = max(1, min(10, final_score))

            escalated = False
            if args.auto_escalate and final_score < args.min_score and model != escalation_model:
                esc_answer, esc_latency = ollama_generate(
                    escalation_model,
                    clipped_prompt,
                    temperature=args.temperature,
                    num_predict=args.answer_max_tokens,
                )
                esc_llm, esc_reason = judge_answer(
                    args.judge_model,
                    task,
                    esc_answer,
                    judge_max_tokens=args.judge_max_tokens,
                    prompt_max_chars=args.prompt_max_chars,
                )
                esc_det, esc_det_reason = deterministic_score(task, esc_answer)
                esc_final = int(round(esc_llm * (1 - args.det_weight) + esc_det * args.det_weight))
                esc_final = max(1, min(10, esc_final))
                if esc_final >= final_score:
                    answer = esc_answer
                    latency += esc_latency
                    llm_score = esc_llm
                    llm_reason = esc_reason
                    det_score = esc_det
                    det_reason = esc_det_reason
                    final_score = esc_final
                    model = escalation_model
                    escalated = True

            token_estimate = approx_tokens(task.get("prompt", "")) + approx_tokens(answer)
            cost = estimate_cost(token_estimate, prices.get(model, prices.get("default", 0.2)))
            baseline_cost = estimate_cost(token_estimate, prices.get(baseline_model, 0.2))

            run = {
                "model": model,
                "score": final_score,
                "llm_score": llm_score,
                "deterministic_score": det_score,
                "judge_reason": llm_reason,
                "deterministic_reason": det_reason,
                "latency_sec": round(latency, 3),
                "approx_tokens": token_estimate,
                "estimated_cost": round(cost, 8),
                "estimated_saving_vs_baseline": round(max(0.0, baseline_cost - cost), 8),
                "escalated": escalated,
                "answer": answer,
            }
            task_out["runs"].append(run)

            scores[model].append(final_score)
            lats[model].append(latency)
            costs[model] += cost
            by_type[ttype][model].append(final_score)
            by_type_lats[ttype][model].append(latency)

        result["tasks"].append(task_out)

    for model in models:
        ms = scores[model]
        if not ms:
            continue
        ls = lats[model]
        result["summary"][model] = {
            "avg_score": round(sum(ms) / len(ms), 3),
            "pass_rate": round(sum(1 for s in ms if s >= args.min_score) / len(ms), 3),
            "avg_latency_sec": round(sum(ls) / len(ls), 3),
            "p50_latency_sec": round(percentile(ls, 50), 3),
            "p95_latency_sec": round(percentile(ls, 95), 3),
            "total_estimated_cost": round(costs[model], 6),
        }

    for ttype, model_scores in by_type.items():
        result["summary_by_type"][ttype] = {}
        for model, vals in model_scores.items():
            lat_vals = by_type_lats[ttype][model]
            result["summary_by_type"][ttype][model] = {
                "avg_score": round(sum(vals) / len(vals), 3),
                "pass_rate": round(sum(1 for s in vals if s >= args.min_score) / len(vals), 3),
                "avg_latency_sec": round(sum(lat_vals) / len(lat_vals), 3),
            }

    for ttype, model_stats in result["summary_by_type"].items():
        candidates = []
        for model, st in model_stats.items():
            if st["avg_score"] >= args.min_score and st["avg_latency_sec"] <= args.sla_max_latency:
                candidates.append((model, st["avg_latency_sec"], st["avg_score"]))

        if candidates:
            candidates.sort(key=lambda x: (x[1], -x[2]))
            result["recommended_routing"][ttype] = candidates[0][0]
        else:
            fallback = sorted(
                model_stats.items(),
                key=lambda kv: (kv[1]["avg_score"], -kv[1]["avg_latency_sec"]),
                reverse=True,
            )
            result["recommended_routing"][ttype] = fallback[0][0]

    write_json(args.out, result)
    print(f"Saved benchmark report: {args.out}")
    return 0


def build_policy(args: argparse.Namespace) -> int:
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    summary = report.get("summary", {})
    if not summary:
        print("No summary in report")
        return 2

    candidates = []
    for model, st in summary.items():
        if st.get("avg_score", 0) >= args.min_score and st.get("avg_latency_sec", 9999) <= args.sla_max_latency:
            candidates.append((model, st.get("avg_latency_sec", 9999), st.get("avg_score", 0)))

    if candidates:
        candidates.sort(key=lambda x: (x[1], -x[2]))
        default_model = candidates[0][0]
    else:
        default_model = max(summary.keys(), key=lambda m: summary[m].get("avg_score", 0))

    escalation_model = max(summary.keys(), key=lambda m: summary[m].get("avg_score", 0))

    routing = report.get("recommended_routing", {})
    for key in ["coding_hard", "coding_light", "summarization", "general"]:
        routing.setdefault(key, default_model)

    policy = {
        "meta": {
            "source_report": args.report,
            "created_at": int(time.time()),
            "min_score": args.min_score,
            "sla_max_latency_sec": args.sla_max_latency,
        },
        "default_model": default_model,
        "escalation_model": escalation_model,
        "retry_policy": {
            "enabled": True,
            "retry_if_score_below": args.min_score,
            "max_retries": 1,
            "retry_model": escalation_model,
        },
        "routing_rules": routing,
        "model_stats": summary,
    }
    write_json(args.out, policy)
    print(f"Saved policy: {args.out}")
    print(f"Recommended default model: {default_model}")
    print(f"Escalation model: {escalation_model}")
    return 0


def route_task(args: argparse.Namespace) -> int:
    policy = json.loads(Path(args.policy).read_text(encoding="utf-8"))
    task_type = args.task_type.strip() if args.task_type else "general"
    selected = policy.get("routing_rules", {}).get(task_type, policy.get("default_model", "qwen3:8b"))
    print(selected)
    return 0


def simulate_production(args: argparse.Namespace) -> int:
    policy = json.loads(Path(args.policy).read_text(encoding="utf-8"))
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    tasks = report.get("tasks", [])
    if not tasks:
        print("No tasks in report")
        return 2

    default_model = policy.get("default_model", "qwen3:8b")
    retry_model = policy.get("retry_policy", {}).get("retry_model", default_model)
    retry_threshold = int(policy.get("retry_policy", {}).get("retry_if_score_below", 7))

    routed = 0
    retried = 0
    avg_scores = []
    latencies = []

    for task in tasks:
        ttype = task.get("type", "general")
        model = policy.get("routing_rules", {}).get(ttype, default_model)
        runs = [r for r in task.get("runs", []) if r.get("model") == model]
        if not runs:
            runs = [r for r in task.get("runs", []) if r.get("model") == default_model]
        if not runs:
            continue
        run = runs[0]
        routed += 1
        score = run.get("score", 1)
        latency = run.get("latency_sec", 0)

        if score < retry_threshold and retry_model != model:
            retry_runs = [r for r in task.get("runs", []) if r.get("model") == retry_model]
            if retry_runs:
                rr = retry_runs[0]
                if rr.get("score", 1) >= score:
                    score = rr.get("score", score)
                    latency += rr.get("latency_sec", 0)
                    retried += 1

        avg_scores.append(score)
        latencies.append(latency)

    sim = {
        "routed_tasks": routed,
        "retried_tasks": retried,
        "avg_score": round(sum(avg_scores) / len(avg_scores), 3) if avg_scores else 0,
        "p95_latency_sec": round(percentile(latencies, 95), 3) if latencies else 0,
    }
    sim["verdict"] = (
        "good"
        if sim["avg_score"] >= 7 and sim["p95_latency_sec"] <= 45
        else "needs-optimization"
    )
    write_json(args.out, sim)
    print(f"Saved production simulation: {args.out}")
    print(
        "Simulation verdict: "
        + sim["verdict"]
        + f", avg_score={sim['avg_score']}, p95={sim['p95_latency_sec']}s"
    )
    return 0


def render_report(args: argparse.Namespace) -> int:
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    summary = report.get("summary", {})
    by_type = report.get("summary_by_type", {})
    routing = report.get("recommended_routing", {})

    cards = []
    for model, st in summary.items():
        cards.append(
            "<div class='card'>"
            f"<h3>{escape(model)}</h3>"
            f"<p><b>Avg Score:</b> {st.get('avg_score', 0)}</p>"
            f"<p><b>Pass Rate:</b> {st.get('pass_rate', 0)}</p>"
            f"<p><b>Avg Latency:</b> {st.get('avg_latency_sec', 0)}s</p>"
            f"<p><b>P50/P95:</b> {st.get('p50_latency_sec', 0)}s / {st.get('p95_latency_sec', 0)}s</p>"
            f"<p><b>Total Cost:</b> ${st.get('total_estimated_cost', 0)}</p>"
            "</div>"
        )

    route_items = "".join(
        f"<li><b>{escape(k)}</b> -> {escape(v)}</li>" for k, v in sorted(routing.items())
    )

    by_type_rows = []
    for ttype, models in sorted(by_type.items()):
        for model, st in models.items():
            by_type_rows.append(
                "<tr>"
                f"<td>{escape(ttype)}</td><td>{escape(model)}</td>"
                f"<td>{st.get('avg_score', '')}</td><td>{st.get('pass_rate', '')}</td><td>{st.get('avg_latency_sec', '')}</td>"
                "</tr>"
            )

    rows = []
    for task in report.get("tasks", []):
        tid = escape(str(task.get("id", "")))
        ttype = escape(str(task.get("type", "")))
        for run in task.get("runs", []):
            rows.append(
                "<tr>"
                f"<td>{tid}</td><td>{ttype}</td><td>{escape(str(run.get('model', '')))}</td>"
                f"<td>{run.get('score', '')}</td><td>{run.get('llm_score', '')}</td><td>{run.get('deterministic_score', '')}</td>"
                f"<td>{run.get('latency_sec', '')}</td><td>{run.get('approx_tokens', '')}</td>"
                f"<td>${run.get('estimated_cost', '')}</td><td>${run.get('estimated_saving_vs_baseline', '')}</td>"
                "</tr>"
            )

    best_model = None
    if summary:
        best_model = max(summary.items(), key=lambda kv: kv[1].get("avg_score", 0))[0]

    verdict = "Insufficient data"
    if best_model:
        m = summary[best_model]
        if m.get("pass_rate", 0) >= 0.9 and m.get("p95_latency_sec", 9999) <= 45:
            verdict = "Launch-ready for pilot"
        elif m.get("pass_rate", 0) >= 0.75:
            verdict = "Usable, optimize latency before wider launch"
        else:
            verdict = "Needs more tuning before launch"

    html = f"""<!doctype html>
<html lang='en'>
<head>
  <meta charset='utf-8' />
  <meta name='viewport' content='width=device-width, initial-scale=1' />
  <title>Model Router Lab Report</title>
  <style>
    :root {{ --bg:#0b1220; --panel:#121c33; --line:#2a3a62; --text:#eaf0ff; --muted:#9fb0d9; }}
    body {{ margin:0; font-family: Segoe UI, sans-serif; color:var(--text); background:radial-gradient(circle at 20% 0%, #1a2d56, #0b1220 45%); }}
    .wrap {{ max-width:1200px; margin:0 auto; padding:24px; }}
    .grid {{ display:grid; gap:12px; grid-template-columns: repeat(auto-fit,minmax(230px,1fr)); }}
    .card {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:12px; }}
    .routing {{ margin-top:14px; background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:12px; }}
    table {{ width:100%; border-collapse:collapse; margin-top:14px; background:var(--panel); border:1px solid var(--line); }}
    th,td {{ border-bottom:1px solid var(--line); padding:8px; text-align:left; font-size:13px; }}
    th {{ background:#1b2a4b; color:#bed0ff; }}
  </style>
</head>
<body>
  <div class='wrap'>
    <h1>Model Router Lab</h1>
    <p>R1 report: dual scoring, SLA routing, and cost-aware recommendations.</p>
        <div class='routing'><h3>Business Verdict</h3><p>{escape(verdict)}</p></div>
    <div class='grid'>{''.join(cards)}</div>
    <div class='routing'>
      <h3>Recommended Routing by Task Type</h3>
      <ul>{route_items}</ul>
    </div>
    <h3>Type-Level Performance</h3>
    <table>
      <thead>
        <tr><th>Task Type</th><th>Model</th><th>Avg Score</th><th>Pass Rate</th><th>Avg Latency(s)</th></tr>
      </thead>
      <tbody>{''.join(by_type_rows)}</tbody>
    </table>
    <h3>Task-Level Runs</h3>
    <table>
      <thead>
        <tr><th>Task</th><th>Type</th><th>Model</th><th>Final</th><th>LLM</th><th>Det</th><th>Latency(s)</th><th>Tokens</th><th>Cost</th><th>Saving</th></tr>
      </thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </div>
</body>
</html>
"""
    write_text(args.out, html)
    print(f"Saved HTML report: {args.out}")
    return 0


def doctor(_: argparse.Namespace) -> int:
    ensure_ollama()
    models = ollama_models()
    print("Ollama reachable")
    print("Installed models:")
    for model in models:
        print(f"- {model}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Model Router Lab: benchmark and routing for local LLMs")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_doc = sub.add_parser("doctor")
    p_doc.set_defaults(func=doctor)

    p_sample = sub.add_parser("sample-tasks")
    p_sample.add_argument("--max-files", type=int, default=20)
    p_sample.add_argument("--out", default="projects/model-router-lab/datasets/repo_tasks.jsonl")
    p_sample.set_defaults(func=sample_tasks_from_repo)

    p_bench = sub.add_parser("benchmark")
    p_bench.add_argument("--dataset", required=True)
    p_bench.add_argument("--models", required=True)
    p_bench.add_argument("--judge-model", default="qwen3:8b")
    p_bench.add_argument("--temperature", type=float, default=0.1)
    p_bench.add_argument("--min-score", type=int, default=7)
    p_bench.add_argument("--det-weight", type=float, default=0.4, help="Weight for deterministic score in final score")
    p_bench.add_argument("--sla-max-latency", type=float, default=45.0, help="SLA max average latency per task type")
    p_bench.add_argument("--prompt-max-chars", type=int, default=1200)
    p_bench.add_argument("--answer-max-tokens", type=int, default=320)
    p_bench.add_argument("--judge-max-tokens", type=int, default=120)
    p_bench.add_argument("--auto-escalate", action="store_true")
    p_bench.add_argument("--prices", default="", help="model=price_per_1m,model2=price")
    p_bench.add_argument("--out", default="projects/model-router-lab/outputs/benchmark.json")
    p_bench.set_defaults(func=run_benchmark)

    p_policy = sub.add_parser("build-policy")
    p_policy.add_argument("--report", required=True)
    p_policy.add_argument("--min-score", type=float, default=7.0)
    p_policy.add_argument("--sla-max-latency", type=float, default=45.0)
    p_policy.add_argument("--out", default="projects/model-router-lab/outputs/policy.json")
    p_policy.set_defaults(func=build_policy)

    p_route = sub.add_parser("route")
    p_route.add_argument("--policy", required=True)
    p_route.add_argument("--task-type", default="general")
    p_route.set_defaults(func=route_task)

    p_render = sub.add_parser("render-report")
    p_render.add_argument("--report", required=True)
    p_render.add_argument("--out", default="projects/model-router-lab/outputs/report.html")
    p_render.set_defaults(func=render_report)

    p_sim = sub.add_parser("simulate-production")
    p_sim.add_argument("--report", required=True)
    p_sim.add_argument("--policy", required=True)
    p_sim.add_argument("--out", default="projects/model-router-lab/outputs/simulation.json")
    p_sim.set_defaults(func=simulate_production)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
