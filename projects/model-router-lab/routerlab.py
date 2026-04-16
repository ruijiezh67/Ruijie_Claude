#!/usr/bin/env python3
import argparse
import ast
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from html import escape
from pathlib import Path

OLLAMA_URL = "http://127.0.0.1:11434"
MINIMAX_API_URL = os.environ.get("MINIMAX_API_URL", "https://api.minimaxi.com/v1/text/chatcompletion_v2")
DEFAULT_PRICES = {
    "qwen3:8b": 0.10,
    "qwen3:32b": 0.45,
}

WORKFLOW_TEMPLATES = {
    "coding_hard": [
        {
            "stage": "planner",
            "agent": "architect",
            "goal": "quality",
            "purpose": "Design approach and identify edge cases before coding.",
        },
        {
            "stage": "builder",
            "agent": "coder",
            "goal": "balanced",
            "purpose": "Implement the change with concrete patch-level output.",
        },
        {
            "stage": "reviewer",
            "agent": "reviewer",
            "goal": "quality",
            "purpose": "Check correctness, regressions, and security risks.",
        },
    ],
    "coding_light": [
        {
            "stage": "builder",
            "agent": "coder",
            "goal": "balanced",
            "purpose": "Apply focused code changes quickly.",
        },
        {
            "stage": "reviewer",
            "agent": "reviewer",
            "goal": "quality",
            "purpose": "Confirm patch quality before merge.",
        },
    ],
    "summarization": [
        {
            "stage": "collector",
            "agent": "analyst",
            "goal": "fast",
            "purpose": "Extract main points and normalize structure.",
        },
        {
            "stage": "editor",
            "agent": "editor",
            "goal": "quality",
            "purpose": "Refine for clarity and decision readiness.",
        },
    ],
    "general": [
        {
            "stage": "solver",
            "agent": "generalist",
            "goal": "balanced",
            "purpose": "Produce practical, directly usable output.",
        }
    ],
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


def minimax_generate(
    model: str,
    prompt: str,
    temperature: float = 0.1,
    num_predict: int = 320,
) -> tuple[str, float]:
    api_key = os.environ.get("MINIMAX_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("MINIMAX_API_KEY is missing")

    remote_model = model.split("/", 1)[1] if "/" in model else model
    payload = {
        "model": remote_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": num_predict,
    }
    req = urllib.request.Request(
        MINIMAX_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    start = time.time()
    with urllib.request.urlopen(req, timeout=900) as resp:
        out = json.loads(resp.read().decode("utf-8"))
    latency = time.time() - start

    text = ""
    choices = out.get("choices") or []
    if choices and isinstance(choices[0], dict):
        msg = choices[0].get("message") or {}
        text = (msg.get("content") or "").strip()
    if not text:
        text = (out.get("reply") or out.get("text") or "").strip()
    if not text:
        raise RuntimeError("MiniMax response does not contain text")
    return text, latency


def is_minimax_model(model: str) -> bool:
    return model.strip().lower().startswith("minimax/")


def model_generate(
    model: str,
    prompt: str,
    temperature: float = 0.1,
    num_predict: int = 320,
) -> tuple[str, float]:
    if is_minimax_model(model):
        return minimax_generate(model, prompt, temperature=temperature, num_predict=num_predict)
    return ollama_generate(model, prompt, temperature=temperature, num_predict=num_predict)


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


def load_json_if_exists(path: str) -> dict | list | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def to_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def cleanup_research_artifacts(topic_slug: str, keep_paths: set[str] | None = None) -> list[str]:
    keep_paths = keep_paths or set()
    removed = []
    patterns = [
        f"projects/model-router-lab/datasets/research_{topic_slug}*.jsonl",
        f"projects/model-router-lab/outputs/benchmark_research_{topic_slug}*.json",
        f"projects/model-router-lab/outputs/policy_research_{topic_slug}*.json",
        f"projects/model-router-lab/outputs/simulation_research_{topic_slug}*.json",
        f"projects/model-router-lab/outputs/control_plane_research_{topic_slug}*.json",
        f"projects/model-router-lab/outputs/control_plane_evolved_research_{topic_slug}*.json",
        f"projects/model-router-lab/outputs/control_plane_evolved_simulation_research_{topic_slug}*.json",
        f"projects/model-router-lab/outputs/report_research_{topic_slug}*.html",
        f"projects/model-router-lab/outputs/research_delta_{topic_slug}*.json",
    ]

    for pattern in patterns:
        for path in Path().glob(pattern):
            path_str = str(path).replace("\\", "/")
            if path_str in keep_paths:
                continue
            try:
                path.unlink()
                removed.append(path_str)
            except FileNotFoundError:
                continue
    return removed


def format_removed_artifacts(removed: list[str]) -> str:
    if not removed:
        return ""
    preview = removed[:5]
    suffix = "" if len(removed) <= 5 else f" ... (+{len(removed) - 5} more)"
    return ", ".join(preview) + suffix


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


def pick_model_from_stats(
    model_stats: dict[str, dict],
    goal: str,
    min_score: float,
    max_latency: float,
) -> str:
    if not model_stats:
        return "qwen3:8b"

    valid = []
    for model, st in model_stats.items():
        score = float(st.get("avg_score", 0))
        latency = float(st.get("avg_latency_sec", 999999))
        pass_rate = float(st.get("pass_rate", 0))
        valid.append((model, score, latency, pass_rate))

    sla_pool = [v for v in valid if v[1] >= min_score and v[2] <= max_latency]
    target = sla_pool if sla_pool else valid

    if goal == "quality":
        target.sort(key=lambda x: (x[1], x[3], -x[2]), reverse=True)
    elif goal == "fast":
        target.sort(key=lambda x: (x[2], -x[1], -x[3]))
    elif goal == "cheap":
        target.sort(key=lambda x: (x[2], -x[1]))
    else:
        # balanced
        target.sort(key=lambda x: ((x[1] * x[3]) / max(0.2, x[2])), reverse=True)

    return target[0][0]


def avg_model_latency(summary: dict[str, dict], model: str) -> float:
    st = summary.get(model, {})
    return float(st.get("avg_latency_sec", 0) or 0)


def avg_model_score(summary: dict[str, dict], model: str) -> float:
    st = summary.get(model, {})
    return float(st.get("avg_score", 0) or 0)


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
    try:
        raw, _ = model_generate(judge_model, prompt, temperature=0.0, num_predict=judge_max_tokens)
    except Exception as exc:
        return 0, f"Judge call failed: {str(exc)[:120]}"
    try:
        m = re.search(r"SCORE\s*:\s*(\d{1,2})", raw, re.IGNORECASE)
        if not m:
            raise ValueError("no-score-tag")
        score = int(m.group(1))
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


def slugify(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower())
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "research"


def build_research_tasks(topic: str) -> list[dict]:
    return [
        {
            "id": "research-1",
            "type": "summarization",
            "prompt": (
                f"Topic: {topic}\\n"
                "You are a research scout. Summarize likely state-of-the-art directions, key baselines, and open problems."
            ),
            "expected": "SOTA, baselines, open problems, evaluation caveats",
            "rubric": "Depth, factual structure, and actionable research framing.",
        },
        {
            "id": "research-2",
            "type": "general",
            "prompt": (
                f"Topic: {topic}\\n"
                "Propose 5 testable hypotheses and rank by novelty, feasibility, and expected impact."
            ),
            "expected": "hypotheses with ranking and measurable criteria",
            "rubric": "Novelty-feasibility tradeoff and scientific clarity.",
        },
        {
            "id": "research-3",
            "type": "coding_hard",
            "prompt": (
                f"Topic: {topic}\\n"
                "Draft a minimal PyTorch experiment scaffold including dataset interface, training loop, and evaluation hooks."
            ),
            "expected": "valid python code block with train/eval scaffold",
            "rubric": "Correctness, modularity, and experiment readiness.",
        },
        {
            "id": "research-4",
            "type": "general",
            "prompt": (
                f"Topic: {topic}\\n"
                "Design an ablation plan with variables, metrics, and stopping criteria."
            ),
            "expected": "variables, metrics, stopping criteria, failure analysis plan",
            "rubric": "Scientific rigor and operational feasibility.",
        },
    ]


def fetch_arxiv_papers(query: str, max_results: int = 8) -> list[dict]:
    q = query.strip()
    if not q:
        return []
    url = (
        "http://export.arxiv.org/api/query?search_query="
        + urllib.parse.quote(q)
        + f"&start=0&max_results={max(1, max_results)}"
    )
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=60) as resp:
        xml_data = resp.read().decode("utf-8", errors="ignore")

    root = ET.fromstring(xml_data)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    papers = []
    for entry in root.findall("atom:entry", ns):
        title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
        summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
        paper_id = (entry.findtext("atom:id", default="", namespaces=ns) or "").strip()
        published = (entry.findtext("atom:published", default="", namespaces=ns) or "").strip()
        authors = []
        for author in entry.findall("atom:author", ns):
            name = (author.findtext("atom:name", default="", namespaces=ns) or "").strip()
            if name:
                authors.append(name)
        if title:
            papers.append(
                {
                    "id": paper_id,
                    "title": title,
                    "summary": summary,
                    "published": published,
                    "authors": authors,
                }
            )
    return papers


def enrich_research_tasks_with_papers(tasks: list[dict], topic: str, papers: list[dict]) -> list[dict]:
    if not papers:
        return tasks

    lines = []
    for idx, p in enumerate(papers, start=1):
        authors = ", ".join(p.get("authors", [])[:3])
        summary = clip_text(p.get("summary", ""), 450).replace("\n", " ")
        lines.append(
            f"[{idx}] {p.get('title', '')}; authors={authors}; year={p.get('published', '')[:4]}; abstract={summary}"
        )
    corpus = "\n".join(lines)

    extra = [
        {
            "id": "research-lit-1",
            "type": "summarization",
            "prompt": (
                f"Topic: {topic}\\n"
                "Use the following arXiv corpus to identify research gaps, conflicting claims, and replication risks.\\n"
                f"CORPUS:\\n{corpus}"
            ),
            "expected": "research gaps, conflicting findings, replication risks",
            "rubric": "Grounded synthesis quality and rigor.",
        },
        {
            "id": "research-lit-2",
            "type": "coding_hard",
            "prompt": (
                f"Topic: {topic}\\n"
                "Based on this arXiv corpus, write a PyTorch-ready experiment plan as code comments + executable scaffold.\\n"
                "Include dataset split, baseline, ablation switches, and metric logging hooks.\\n"
                f"CORPUS:\\n{corpus}"
            ),
            "expected": "executable scaffold with baselines and ablations",
            "rubric": "Engineering readiness and scientific traceability.",
        },
    ]
    return tasks + extra


def fetch_arxiv_cmd(args: argparse.Namespace) -> int:
    papers = fetch_arxiv_papers(args.query, max_results=args.max_results)
    payload = {
        "query": args.query,
        "max_results": args.max_results,
        "fetched": len(papers),
        "papers": papers,
    }
    write_json(args.out, payload)
    print(f"Saved arXiv results: {args.out}")
    print(f"Paper count: {len(papers)}")
    return 0


def make_research_dataset(args: argparse.Namespace) -> int:
    tasks = build_research_tasks(args.topic)
    papers = []
    arxiv_query = (getattr(args, "arxiv_query", "") or "").strip()
    arxiv_max_results = int(getattr(args, "arxiv_max_results", 8) or 8)
    if arxiv_query:
        try:
            papers = fetch_arxiv_papers(arxiv_query, max_results=arxiv_max_results)
            tasks = enrich_research_tasks_with_papers(tasks, args.topic, papers)
        except Exception as exc:
            print(f"Warning: arXiv fetch failed, continuing without papers: {str(exc)[:160]}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(t, ensure_ascii=False) for t in tasks) + "\n", encoding="utf-8")
    print(f"Saved research dataset: {args.out}")
    print(f"Task count: {len(tasks)}")
    if papers:
        print(f"Included arXiv papers: {len(papers)}")
    return 0


def run_benchmark(args: argparse.Namespace) -> int:
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if any(not is_minimax_model(m) for m in models) or not is_minimax_model(args.judge_model):
        ensure_ollama()
    tasks = load_tasks(args.dataset)
    if not tasks:
        print("Dataset is empty")
        return 2

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
    model_failures = defaultdict(int)

    for task in tasks:
        ttype = task.get("type", "general")
        task_out = {"id": task.get("id"), "type": ttype, "runs": []}

        clipped_prompt = clip_text(task.get("prompt", ""), args.prompt_max_chars)
        for model in ordered_models:
            try:
                answer, latency = model_generate(
                    model,
                    clipped_prompt,
                    temperature=args.temperature,
                    num_predict=args.answer_max_tokens,
                )
            except Exception as exc:
                model_failures[model] += 1
                task_out["runs"].append(
                    {
                        "model": model,
                        "score": 0,
                        "llm_score": 0,
                        "deterministic_score": 0,
                        "judge_reason": "model-call-failed",
                        "deterministic_reason": "model-call-failed",
                        "latency_sec": 0,
                        "approx_tokens": 0,
                        "estimated_cost": 0,
                        "estimated_saving_vs_baseline": 0,
                        "escalated": False,
                        "answer": "",
                        "error": str(exc)[:200],
                    }
                )
                continue
            if not answer.strip():
                try:
                    answer, retry_latency = model_generate(
                        model,
                        task.get("prompt", ""),
                        temperature=max(0.05, args.temperature),
                        num_predict=max(320, args.answer_max_tokens),
                    )
                    latency += retry_latency
                except Exception as exc:
                    model_failures[model] += 1
                    task_out["runs"].append(
                        {
                            "model": model,
                            "score": 0,
                            "llm_score": 0,
                            "deterministic_score": 0,
                            "judge_reason": "model-retry-failed",
                            "deterministic_reason": "model-retry-failed",
                            "latency_sec": round(latency, 3),
                            "approx_tokens": 0,
                            "estimated_cost": 0,
                            "estimated_saving_vs_baseline": 0,
                            "escalated": False,
                            "answer": "",
                            "error": str(exc)[:200],
                        }
                    )
                    continue
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
                try:
                    esc_answer, esc_latency = model_generate(
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
                except Exception:
                    model_failures[escalation_model] += 1

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

    if model_failures:
        result["meta"]["model_failures"] = dict(model_failures)

    write_json(args.out, result)
    print(f"Saved benchmark report: {args.out}")
    if model_failures:
        print("Model failures (non-fatal): " + json.dumps(dict(model_failures), ensure_ascii=False))
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


def build_control_plane(args: argparse.Namespace) -> int:
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    policy = json.loads(Path(args.policy).read_text(encoding="utf-8"))
    summary = report.get("summary", {})
    by_type = report.get("summary_by_type", {})
    routing_rules = policy.get("routing_rules", {})
    default_model = policy.get("default_model", "qwen3:8b")
    escalation_model = policy.get("escalation_model", default_model)

    workflows = {}
    task_types = sorted(set(list(WORKFLOW_TEMPLATES.keys()) + list(routing_rules.keys()) + list(by_type.keys())))

    for ttype in task_types:
        stage_templates = WORKFLOW_TEMPLATES.get(ttype, WORKFLOW_TEMPLATES["general"])
        model_stats = by_type.get(ttype, summary)
        primary = routing_rules.get(ttype, default_model)
        stages = []
        predicted_latency = 0.0
        predicted_score = 0.0

        for idx, template in enumerate(stage_templates):
            goal = template.get("goal", "balanced")
            picked = pick_model_from_stats(
                model_stats,
                goal=goal,
                min_score=args.min_score,
                max_latency=args.max_latency,
            )
            if idx == 0 and primary in model_stats:
                picked = primary

            stage_latency = avg_model_latency(summary, picked)
            stage_score = avg_model_score(summary, picked)
            predicted_latency += stage_latency
            predicted_score += stage_score

            stages.append(
                {
                    "name": template["stage"],
                    "agent": template["agent"],
                    "purpose": template["purpose"],
                    "model": picked,
                    "fallback_model": escalation_model,
                    "goal": goal,
                    "predicted_latency_sec": round(stage_latency, 3),
                    "predicted_score": round(stage_score, 3),
                }
            )

        stage_count = max(1, len(stages))
        workflow_avg_score = predicted_score / stage_count
        escalation_trigger = (
            "stage_score_below_threshold"
            if workflow_avg_score < args.min_score
            else "final_quality_gate_fail"
        )

        workflows[ttype] = {
            "task_type": ttype,
            "default_model": primary,
            "workflow_mode": "multi-agent",
            "predicted_total_latency_sec": round(predicted_latency, 3),
            "predicted_avg_score": round(workflow_avg_score, 3),
            "escalation_trigger": escalation_trigger,
            "stages": stages,
            "quality_gates": {
                "min_stage_score": args.min_score,
                "max_total_latency_sec": args.max_latency * max(1, len(stages)),
                "require_reviewer_for_coding": ttype.startswith("coding"),
            },
        }

    cp = {
        "meta": {
            "source_report": args.report,
            "source_policy": args.policy,
            "created_at": int(time.time()),
            "type": "autonomous-ai-engineering-control-plane",
            "version": "r1",
        },
        "global_defaults": {
            "fallback_model": escalation_model,
            "retry": {"max_retries": 1, "backoff_ms": 0},
            "routing_strategy": "task-type + stage-goal",
        },
        "workflows": workflows,
    }

    write_json(args.out, cp)
    print(f"Saved control-plane workflow policy: {args.out}")
    print(f"Workflow count: {len(workflows)}")
    return 0


def route_workflow(args: argparse.Namespace) -> int:
    cp = json.loads(Path(args.control_plane).read_text(encoding="utf-8"))
    workflows = cp.get("workflows", {})
    ttype = (args.task_type or "general").strip()

    wf = workflows.get(ttype)
    if not wf:
        wf = workflows.get("general")
    if not wf:
        print("{}")
        return 0

    stage_plan = list(wf.get("stages", []))
    if args.mode == "quick" and len(stage_plan) > 1:
        stage_plan = stage_plan[:-1]
    elif args.mode == "strict" and ttype.startswith("coding"):
        has_reviewer = any(s.get("agent") == "reviewer" for s in stage_plan)
        if not has_reviewer:
            stage_plan.append(
                {
                    "name": "reviewer",
                    "agent": "reviewer",
                    "purpose": "Strict verification for correctness and risk.",
                    "model": cp.get("global_defaults", {}).get("fallback_model", "qwen3:8b"),
                    "fallback_model": cp.get("global_defaults", {}).get("fallback_model", "qwen3:8b"),
                    "goal": "quality",
                    "predicted_latency_sec": 0,
                    "predicted_score": 0,
                }
            )

    plan = {
        "task_type": ttype,
        "mode": args.mode,
        "default_model": wf.get("default_model"),
        "escalation_trigger": wf.get("escalation_trigger"),
        "stages": stage_plan,
    }
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    return 0


def _workflow_stage_signature(stages: list[dict]) -> str:
    return "|".join(f"{s.get('name')}:{s.get('model')}:{s.get('agent')}" for s in stages)


def _simulate_stages_on_task(task: dict, stages: list[dict]) -> tuple[float, float]:
    task_runs = task.get("runs", [])
    by_model = {r.get("model"): r for r in task_runs}

    total_latency = 0.0
    stage_scores = []
    for st in stages:
        model = st.get("model")
        run = by_model.get(model)
        if run:
            total_latency += float(run.get("latency_sec", 0) or 0)
            stage_scores.append(float(run.get("score", 0) or 0))
        else:
            total_latency += float(st.get("predicted_latency_sec", 0) or 0)
            stage_scores.append(float(st.get("predicted_score", 0) or 0))

    avg_score = sum(stage_scores) / max(1, len(stage_scores))
    return avg_score, total_latency


def _evaluate_stage_plan(tasks: list[dict], stages: list[dict], min_score: float, max_latency: float) -> dict:
    scores = []
    lats = []
    for task in tasks:
        s, l = _simulate_stages_on_task(task, stages)
        scores.append(s)
        lats.append(l)

    avg_score = round(sum(scores) / len(scores), 3) if scores else 0.0
    p95_latency = round(percentile(lats, 95), 3) if lats else 0.0
    stage_count = len(stages)
    feasible = avg_score >= min_score and p95_latency <= max_latency
    # Strongly prefer feasible plans, then quality, then lower latency and fewer stages.
    utility = (
        (10000.0 if feasible else 0.0)
        + (avg_score * 100.0)
        - p95_latency
        - (stage_count * 2.0)
    )
    return {
        "avg_score": avg_score,
        "p95_latency_sec": p95_latency,
        "stage_count": stage_count,
        "feasible": feasible,
        "utility": round(utility, 3),
    }


def _candidate_stage_sets(ttype: str, stages: list[dict]) -> list[list[dict]]:
    cands = [list(stages)]
    if len(stages) > 1:
        cands.append(list(stages[:-1]))

    if ttype.startswith("coding"):
        builder = next((s for s in stages if s.get("agent") == "coder"), None)
        reviewer = next((s for s in stages if s.get("agent") == "reviewer"), None)
        planner = next((s for s in stages if s.get("agent") == "architect"), None)
        if builder and reviewer:
            cands.append([builder, reviewer])
        if planner and builder:
            cands.append([planner, builder])
        if builder:
            cands.append([builder])
    else:
        if stages:
            cands.append([stages[0]])
        if len(stages) >= 2:
            cands.append([stages[0], stages[-1]])

    uniq = []
    seen = set()
    for st in cands:
        if not st:
            continue
        sig = _workflow_stage_signature(st)
        if sig in seen:
            continue
        seen.add(sig)
        uniq.append(st)
    return uniq


def evolve_control_plane(args: argparse.Namespace) -> int:
    cp = json.loads(Path(args.control_plane).read_text(encoding="utf-8"))
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    tasks = report.get("tasks", [])
    workflows = cp.get("workflows", {})
    if not tasks or not workflows:
        print("Control plane or report has no tasks/workflows")
        return 2

    tasks_by_type = defaultdict(list)
    for task in tasks:
        tasks_by_type[task.get("type", "general")].append(task)

    evolved = json.loads(json.dumps(cp))
    evolved["meta"]["version"] = "r2-evolved"
    evolved["meta"]["evolved_at"] = int(time.time())
    evolved["meta"]["source_control_plane"] = args.control_plane
    evolved.setdefault("evolution", {"iterations": [], "final_selection": {}})

    for i in range(max(1, args.max_iters)):
        iter_log = {"iteration": i + 1, "changes": []}
        improved_any = False

        for ttype, wf in list(evolved.get("workflows", {}).items()):
            relevant_tasks = tasks_by_type.get(ttype) or tasks_by_type.get("general") or tasks
            base_stages = list(wf.get("stages", []))
            candidates = _candidate_stage_sets(ttype, base_stages)
            scored = []
            for cand in candidates:
                ev = _evaluate_stage_plan(relevant_tasks, cand, args.min_score, args.max_latency)
                scored.append((ev, cand))

            if not scored:
                continue
            scored.sort(key=lambda x: x[0]["utility"], reverse=True)
            best_ev, best_stages = scored[0]
            cur_ev = _evaluate_stage_plan(relevant_tasks, base_stages, args.min_score, args.max_latency)

            if best_ev["utility"] > cur_ev["utility"]:
                improved_any = True
                wf["stages"] = best_stages
                wf["predicted_total_latency_sec"] = best_ev["p95_latency_sec"]
                wf["predicted_avg_score"] = best_ev["avg_score"]
                wf["quality_gates"]["max_total_latency_sec"] = args.max_latency
                iter_log["changes"].append(
                    {
                        "task_type": ttype,
                        "from_stage_count": len(base_stages),
                        "to_stage_count": len(best_stages),
                        "score": best_ev["avg_score"],
                        "p95_latency_sec": best_ev["p95_latency_sec"],
                        "feasible": best_ev["feasible"],
                    }
                )

            evolved["evolution"]["final_selection"][ttype] = {
                "stage_count": len(wf.get("stages", [])),
                "stages": [s.get("name") for s in wf.get("stages", [])],
                "models": [s.get("model") for s in wf.get("stages", [])],
            }

        evolved["evolution"]["iterations"].append(iter_log)
        if not improved_any:
            break

    write_json(args.out, evolved)
    print(f"Saved evolved control plane: {args.out}")
    print(f"Iterations logged: {len(evolved.get('evolution', {}).get('iterations', []))}")
    return 0


def simulate_control_plane(args: argparse.Namespace) -> int:
    cp = json.loads(Path(args.control_plane).read_text(encoding="utf-8"))
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    tasks = report.get("tasks", [])
    workflows = cp.get("workflows", {})
    if not tasks:
        print("No tasks in report")
        return 2

    pred_scores = []
    pred_lats = []
    escalations = 0
    routed = 0

    for task in tasks:
        ttype = task.get("type", "general")
        wf = workflows.get(ttype) or workflows.get("general")
        if not wf:
            continue
        routed += 1

        task_runs = task.get("runs", [])
        by_model = {r.get("model"): r for r in task_runs}

        total_latency = 0.0
        stage_scores = []
        for st in wf.get("stages", []):
            model = st.get("model")
            run = by_model.get(model)
            if run:
                total_latency += float(run.get("latency_sec", 0) or 0)
                stage_scores.append(float(run.get("score", 0) or 0))
            else:
                total_latency += float(st.get("predicted_latency_sec", 0) or 0)
                stage_scores.append(float(st.get("predicted_score", 0) or 0))

        avg_score = sum(stage_scores) / max(1, len(stage_scores))
        if avg_score < args.min_score:
            escalations += 1

        pred_scores.append(avg_score)
        pred_lats.append(total_latency)

    out = {
        "routed_tasks": routed,
        "escalation_candidates": escalations,
        "predicted_avg_score": round(sum(pred_scores) / len(pred_scores), 3) if pred_scores else 0,
        "predicted_p95_latency_sec": round(percentile(pred_lats, 95), 3) if pred_lats else 0,
    }
    out["verdict"] = (
        "innovative-ready"
        if out["predicted_avg_score"] >= args.min_score and out["predicted_p95_latency_sec"] <= args.max_latency
        else "needs-more-policy-evolution"
    )

    write_json(args.out, out)
    print(f"Saved control-plane simulation: {args.out}")
    print(
        "Control-plane verdict: "
        + out["verdict"]
        + f", avg_score={out['predicted_avg_score']}, p95={out['predicted_p95_latency_sec']}s"
    )
    return 0


def run_research_cycle(args: argparse.Namespace) -> int:
    topic_slug = slugify(args.topic)
    run_id_raw = (args.run_id or "").strip()
    run_id = slugify(run_id_raw)
    run_suffix = f"_{run_id}" if run_id else ""

    dataset_path = f"projects/model-router-lab/datasets/research_{topic_slug}{run_suffix}.jsonl"
    benchmark_path = f"projects/model-router-lab/outputs/benchmark_research_{topic_slug}{run_suffix}.json"
    policy_path = f"projects/model-router-lab/outputs/policy_research_{topic_slug}{run_suffix}.json"
    report_path = f"projects/model-router-lab/outputs/report_research_{topic_slug}{run_suffix}.html"
    cp_path = f"projects/model-router-lab/outputs/control_plane_research_{topic_slug}{run_suffix}.json"
    cp_evolved_path = f"projects/model-router-lab/outputs/control_plane_evolved_research_{topic_slug}{run_suffix}.json"
    sim_path = f"projects/model-router-lab/outputs/simulation_research_{topic_slug}{run_suffix}.json"
    cp_sim_path = (
        f"projects/model-router-lab/outputs/control_plane_evolved_simulation_research_{topic_slug}{run_suffix}.json"
    )
    history_path = f"projects/model-router-lab/outputs/research_history_{topic_slug}.json"
    delta_path = f"projects/model-router-lab/outputs/research_delta_{topic_slug}{run_suffix}.json"

    if not getattr(args, "keep_previous_artifacts", False):
        removed = cleanup_research_artifacts(
            topic_slug,
            keep_paths={history_path, delta_path},
        )
        if removed:
            preview = format_removed_artifacts(removed)
            print(f"Cleaned previous artifacts: {len(removed)}")
            if preview:
                print(f"Removed: {preview}")
    else:
        print("Skipped cleanup of previous artifacts")

    rc = make_research_dataset(
        argparse.Namespace(
            topic=args.topic,
            out=dataset_path,
            arxiv_query=args.arxiv_query,
            arxiv_max_results=args.arxiv_max_results,
        )
    )
    if rc != 0:
        return rc

    rc = run_benchmark(
        argparse.Namespace(
            dataset=dataset_path,
            models=args.models,
            judge_model=args.judge_model,
            temperature=args.temperature,
            min_score=args.min_score,
            det_weight=args.det_weight,
            sla_max_latency=args.sla_max_latency,
            prompt_max_chars=args.prompt_max_chars,
            answer_max_tokens=args.answer_max_tokens,
            judge_max_tokens=args.judge_max_tokens,
            auto_escalate=args.auto_escalate,
            prices=args.prices,
            out=benchmark_path,
        )
    )
    if rc != 0:
        return rc

    rc = build_policy(
        argparse.Namespace(
            report=benchmark_path,
            min_score=args.min_score,
            sla_max_latency=args.sla_max_latency,
            out=policy_path,
        )
    )
    if rc != 0:
        return rc

    rc = simulate_production(
        argparse.Namespace(
            report=benchmark_path,
            policy=policy_path,
            out=sim_path,
        )
    )
    if rc != 0:
        return rc

    rc = build_control_plane(
        argparse.Namespace(
            report=benchmark_path,
            policy=policy_path,
            min_score=args.min_score,
            max_latency=args.sla_max_latency,
            out=cp_path,
        )
    )
    if rc != 0:
        return rc

    rc = evolve_control_plane(
        argparse.Namespace(
            control_plane=cp_path,
            report=benchmark_path,
            min_score=args.min_score,
            max_latency=args.sla_max_latency,
            max_iters=args.max_evolve_iters,
            out=cp_evolved_path,
        )
    )
    if rc != 0:
        return rc

    rc = simulate_control_plane(
        argparse.Namespace(
            control_plane=cp_evolved_path,
            report=benchmark_path,
            min_score=args.min_score,
            max_latency=args.sla_max_latency,
            out=cp_sim_path,
        )
    )
    if rc != 0:
        return rc

    rc = render_report(
        argparse.Namespace(
            report=benchmark_path,
            out=report_path,
        )
    )
    if rc != 0:
        return rc

    sim = load_json_if_exists(sim_path) or {}
    cp_sim = load_json_if_exists(cp_sim_path) or {}
    history_doc = load_json_if_exists(history_path)
    runs = []
    if isinstance(history_doc, dict) and isinstance(history_doc.get("runs"), list):
        runs = history_doc.get("runs", [])

    previous_run = runs[-1] if runs else None
    run_entry = {
        "topic": args.topic,
        "topic_slug": topic_slug,
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": {
            "dataset": dataset_path,
            "benchmark": benchmark_path,
            "policy": policy_path,
            "simulation": sim_path,
            "control_plane": cp_path,
            "control_plane_evolved": cp_evolved_path,
            "control_plane_simulation": cp_sim_path,
            "report": report_path,
        },
        "metrics": {
            "simulation": {
                "avg_score": to_float(sim.get("avg_score", 0)),
                "p95_latency_sec": to_float(sim.get("p95_latency_sec", 0)),
                "verdict": sim.get("verdict", ""),
            },
            "control_plane": {
                "avg_score": to_float(cp_sim.get("predicted_avg_score", 0)),
                "p95_latency_sec": to_float(cp_sim.get("predicted_p95_latency_sec", 0)),
                "verdict": cp_sim.get("verdict", ""),
            },
        },
    }
    runs.append(run_entry)

    write_json(
        history_path,
        {
            "topic": args.topic,
            "topic_slug": topic_slug,
            "runs": runs,
        },
    )

    delta = {
        "topic": args.topic,
        "topic_slug": topic_slug,
        "current_run": run_entry,
        "has_previous": bool(previous_run),
    }
    if isinstance(previous_run, dict):
        prev_sim = ((previous_run.get("metrics") or {}).get("simulation") or {})
        prev_cp = ((previous_run.get("metrics") or {}).get("control_plane") or {})
        delta["previous_run"] = {
            "run_id": previous_run.get("run_id", ""),
            "timestamp": previous_run.get("timestamp", ""),
        }
        delta["changes"] = {
            "simulation": {
                "avg_score_delta": round(to_float(run_entry["metrics"]["simulation"]["avg_score"]) - to_float(prev_sim.get("avg_score", 0)), 3),
                "p95_latency_sec_delta": round(to_float(run_entry["metrics"]["simulation"]["p95_latency_sec"]) - to_float(prev_sim.get("p95_latency_sec", 0)), 3),
                "verdict_changed": run_entry["metrics"]["simulation"]["verdict"] != (prev_sim.get("verdict", "")),
            },
            "control_plane": {
                "avg_score_delta": round(to_float(run_entry["metrics"]["control_plane"]["avg_score"]) - to_float(prev_cp.get("avg_score", 0)), 3),
                "p95_latency_sec_delta": round(to_float(run_entry["metrics"]["control_plane"]["p95_latency_sec"]) - to_float(prev_cp.get("p95_latency_sec", 0)), 3),
                "verdict_changed": run_entry["metrics"]["control_plane"]["verdict"] != (prev_cp.get("verdict", "")),
            },
        }

    write_json(delta_path, delta)
    print(f"Saved research history: {history_path}")
    print(f"Saved research delta: {delta_path}")

    summary = {
        "topic": args.topic,
        "run_id": run_id,
        "arxiv_query": args.arxiv_query,
        "dataset": dataset_path,
        "benchmark": benchmark_path,
        "policy": policy_path,
        "simulation": sim_path,
        "control_plane": cp_path,
        "control_plane_evolved": cp_evolved_path,
        "control_plane_simulation": cp_sim_path,
        "report": report_path,
        "history": history_path,
        "delta": delta_path,
    }
    print("Research cycle completed")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
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


def create_digital_person(args: argparse.Namespace) -> int:
    person_id = slugify(args.name)
    profile_path = args.out or f"projects/model-router-lab/digital_people/{person_id}.json"
    goals = [g.strip() for g in (args.goals or "").split(",") if g.strip()]
    constraints = [c.strip() for c in (args.constraints or "").split(",") if c.strip()]

    profile = {
        "id": person_id,
        "name": args.name,
        "role": args.role,
        "tone": args.tone,
        "domain": args.domain,
        "goals": goals,
        "constraints": constraints,
        "default_model": args.model,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(profile_path, profile)
    print(f"Saved digital person profile: {profile_path}")
    return 0


def build_digital_person_prompt(profile: dict, memory_turns: list[dict], user_message: str) -> str:
    name = str(profile.get("name", "Digital Person"))
    role = str(profile.get("role", "research coworker"))
    tone = str(profile.get("tone", "clear, practical, and concise"))
    domain = str(profile.get("domain", "general"))
    goals = profile.get("goals") if isinstance(profile.get("goals"), list) else []
    constraints = profile.get("constraints") if isinstance(profile.get("constraints"), list) else []

    goals_text = "\\n".join(f"- {g}" for g in goals) if goals else "- Deliver useful and actionable output"
    constraints_text = "\\n".join(f"- {c}" for c in constraints) if constraints else "- Be factual and avoid fabricated claims"

    history_parts = []
    for turn in memory_turns:
        if not isinstance(turn, dict):
            continue
        u = str(turn.get("user", "")).strip()
        a = str(turn.get("assistant", "")).strip()
        if u:
            history_parts.append(f"User: {u}")
        if a:
            history_parts.append(f"{name}: {a}")
    history_block = "\\n".join(history_parts[-12:]) if history_parts else "(no prior history)"

    return (
        f"You are {name}, a digital person.\\n"
        f"Role: {role}\\n"
        f"Tone: {tone}\\n"
        f"Domain focus: {domain}\\n"
        "Goals:\\n"
        f"{goals_text}\\n"
        "Constraints:\\n"
        f"{constraints_text}\\n\\n"
        "Conversation history:\\n"
        f"{history_block}\\n\\n"
        "Respond with: 1) short answer, 2) concrete next actions.\\n"
        f"User: {user_message}"
    )


def chat_with_digital_person(args: argparse.Namespace) -> int:
    profile_doc = load_json_if_exists(args.profile)
    if not isinstance(profile_doc, dict):
        print(f"Invalid profile: {args.profile}")
        return 2

    person_id = str(profile_doc.get("id") or slugify(str(profile_doc.get("name", "digital-person"))))
    memory_path = args.memory or f"projects/model-router-lab/outputs/digital_person_memory_{person_id}.json"
    memory_doc = load_json_if_exists(memory_path)
    if isinstance(memory_doc, list):
        turns = memory_doc
    elif isinstance(memory_doc, dict) and isinstance(memory_doc.get("turns"), list):
        turns = memory_doc.get("turns", [])
    else:
        turns = []

    keep_turns = max(0, int(args.keep_turns))
    selected_history = turns[-keep_turns:] if keep_turns > 0 else []
    prompt = build_digital_person_prompt(profile_doc, selected_history, args.message)

    model = args.model or str(profile_doc.get("default_model", "qwen3:8b"))
    try:
        answer, latency = model_generate(
            model,
            prompt,
            temperature=float(args.temperature),
            num_predict=int(args.max_tokens),
        )
    except Exception as exc:
        print(f"digital-person inference failed: {exc}")
        return 2

    turn = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": model,
        "latency_sec": round(latency, 3),
        "user": args.message,
        "assistant": answer,
    }
    turns.append(turn)
    write_json(memory_path, {"turns": turns} if args.memory_format == "object" else turns)

    output = {
        "profile": args.profile,
        "memory": memory_path,
        "model": model,
        "latency_sec": round(latency, 3),
        "answer": answer,
    }
    if args.out:
        write_json(args.out, output)
        print(f"Saved digital person response: {args.out}")

    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


def doctor(_: argparse.Namespace) -> int:
    try:
        ensure_ollama()
        models = ollama_models()
        print("Ollama reachable")
        print("Installed models:")
        for model in models:
            print(f"- {model}")
    except Exception as exc:
        print("Ollama unavailable: " + str(exc))
    mm_key = os.environ.get("MINIMAX_API_KEY", "").strip()
    print("MiniMax API key: " + ("configured" if mm_key else "missing"))
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

    p_research_ds = sub.add_parser("make-research-dataset")
    p_research_ds.add_argument("--topic", required=True)
    p_research_ds.add_argument("--arxiv-query", default="", help="Optional arXiv query to enrich dataset")
    p_research_ds.add_argument("--arxiv-max-results", type=int, default=8)
    p_research_ds.add_argument("--out", default="projects/model-router-lab/datasets/research_tasks.jsonl")
    p_research_ds.set_defaults(func=make_research_dataset)

    p_arxiv = sub.add_parser("fetch-arxiv")
    p_arxiv.add_argument("--query", required=True)
    p_arxiv.add_argument("--max-results", type=int, default=8)
    p_arxiv.add_argument("--out", default="projects/model-router-lab/outputs/arxiv_results.json")
    p_arxiv.set_defaults(func=fetch_arxiv_cmd)

    p_create_person = sub.add_parser("create-digital-person")
    p_create_person.add_argument("--name", required=True)
    p_create_person.add_argument("--role", default="AI research coworker")
    p_create_person.add_argument("--tone", default="clear, practical, and proactive")
    p_create_person.add_argument("--domain", default="applied AI research and engineering")
    p_create_person.add_argument("--goals", default="Deliver useful ideas,Propose executable experiments,Track progress")
    p_create_person.add_argument("--constraints", default="Be factual,Avoid unsupported claims,Keep outputs actionable")
    p_create_person.add_argument("--model", default="qwen3:8b")
    p_create_person.add_argument("--out", default="")
    p_create_person.set_defaults(func=create_digital_person)

    p_chat_person = sub.add_parser("chat-digital-person")
    p_chat_person.add_argument("--profile", required=True)
    p_chat_person.add_argument("--message", required=True)
    p_chat_person.add_argument("--model", default="")
    p_chat_person.add_argument("--temperature", type=float, default=0.2)
    p_chat_person.add_argument("--max-tokens", type=int, default=320)
    p_chat_person.add_argument("--keep-turns", type=int, default=6)
    p_chat_person.add_argument("--memory", default="")
    p_chat_person.add_argument("--memory-format", choices=["list", "object"], default="list")
    p_chat_person.add_argument("--out", default="")
    p_chat_person.set_defaults(func=chat_with_digital_person)

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

    p_cp = sub.add_parser("build-control-plane")
    p_cp.add_argument("--report", required=True)
    p_cp.add_argument("--policy", required=True)
    p_cp.add_argument("--min-score", type=float, default=7.0)
    p_cp.add_argument("--max-latency", type=float, default=45.0)
    p_cp.add_argument("--out", default="projects/model-router-lab/outputs/control_plane.json")
    p_cp.set_defaults(func=build_control_plane)

    p_route_wf = sub.add_parser("route-workflow")
    p_route_wf.add_argument("--control-plane", required=True)
    p_route_wf.add_argument("--task-type", default="general")
    p_route_wf.add_argument("--mode", choices=["standard", "quick", "strict"], default="standard")
    p_route_wf.set_defaults(func=route_workflow)

    p_sim_cp = sub.add_parser("simulate-control-plane")
    p_sim_cp.add_argument("--control-plane", required=True)
    p_sim_cp.add_argument("--report", required=True)
    p_sim_cp.add_argument("--min-score", type=float, default=7.0)
    p_sim_cp.add_argument("--max-latency", type=float, default=45.0)
    p_sim_cp.add_argument("--out", default="projects/model-router-lab/outputs/control_plane_simulation.json")
    p_sim_cp.set_defaults(func=simulate_control_plane)

    p_research = sub.add_parser("run-research-cycle")
    p_research.add_argument("--topic", required=True)
    p_research.add_argument("--models", required=True, help="comma separated models, e.g. minimax/MiniMax-M2.7,qwen3:8b")
    p_research.add_argument("--judge-model", default="qwen3:8b")
    p_research.add_argument("--temperature", type=float, default=0.1)
    p_research.add_argument("--min-score", type=int, default=7)
    p_research.add_argument("--det-weight", type=float, default=0.4)
    p_research.add_argument("--sla-max-latency", type=float, default=45.0)
    p_research.add_argument("--prompt-max-chars", type=int, default=1200)
    p_research.add_argument("--answer-max-tokens", type=int, default=320)
    p_research.add_argument("--judge-max-tokens", type=int, default=120)
    p_research.add_argument("--auto-escalate", action="store_true")
    p_research.add_argument("--prices", default="")
    p_research.add_argument("--max-evolve-iters", type=int, default=3)
    p_research.add_argument("--arxiv-query", default="", help="Optional arXiv query for paper-grounded tasks")
    p_research.add_argument("--arxiv-max-results", type=int, default=8)
    p_research.add_argument("--run-id", default="", help="Optional run identifier, e.g. 20260415_nightly")
    p_research.add_argument("--keep-previous-artifacts", action="store_true", help="Keep prior same-topic research outputs instead of cleaning them first")
    p_research.set_defaults(func=run_research_cycle)

    p_evolve_cp = sub.add_parser("evolve-control-plane")
    p_evolve_cp.add_argument("--control-plane", required=True)
    p_evolve_cp.add_argument("--report", required=True)
    p_evolve_cp.add_argument("--min-score", type=float, default=7.0)
    p_evolve_cp.add_argument("--max-latency", type=float, default=45.0)
    p_evolve_cp.add_argument("--max-iters", type=int, default=3)
    p_evolve_cp.add_argument("--out", default="projects/model-router-lab/outputs/control_plane_evolved.json")
    p_evolve_cp.set_defaults(func=evolve_control_plane)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
