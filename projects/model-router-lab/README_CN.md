# Model Router Lab

这是一个继续在你现有 Claude Code 工作区上开发的项目，目标不是聊天壳子，而是：

- 用真实仓库任务做评测
- 在质量阈值和 SLA 时延阈值下自动路由模型
- 双重评分: LLM Judge + deterministic checks
- 给出成本估算与相对基线节省
- 产出可视化报告，方便发 GitHub / 社媒

## 快速开始

1. 检查本地模型

```bash
python projects/model-router-lab/routerlab.py doctor
```

2. 从当前仓库自动抽样任务

```bash
python projects/model-router-lab/routerlab.py sample-tasks \
  --max-files 12 \
  --out projects/model-router-lab/datasets/repo_tasks.jsonl
```

3. 跑评测（支持自定义价格）

```bash
python projects/model-router-lab/routerlab.py benchmark \
  --dataset projects/model-router-lab/datasets/repo_tasks.jsonl \
  --models qwen3:8b,qwen3:32b \
  --judge-model qwen3:8b \
  --min-score 7 \
  --det-weight 0.4 \
  --sla-max-latency 45 \
  --prices qwen3:8b=0.10,qwen3:32b=0.45 \
  --out projects/model-router-lab/outputs/benchmark.json
```

4. 生成策略

```bash
python projects/model-router-lab/routerlab.py build-policy \
  --report projects/model-router-lab/outputs/benchmark.json \
  --min-score 7 \
  --sla-max-latency 45 \
  --out projects/model-router-lab/outputs/policy.json
```

5. 渲染可视化报告

```bash
python projects/model-router-lab/routerlab.py render-report \
  --report projects/model-router-lab/outputs/benchmark.json \
  --out projects/model-router-lab/outputs/report.html
```

6. 生成多代理控制平面（创新模式）

```bash
python projects/model-router-lab/routerlab.py build-control-plane \
  --report projects/model-router-lab/outputs/benchmark.json \
  --policy projects/model-router-lab/outputs/policy.json \
  --min-score 7 \
  --max-latency 45 \
  --out projects/model-router-lab/outputs/control_plane.json
```

7. 查看某类任务的工作流路由

```bash
python projects/model-router-lab/routerlab.py route-workflow \
  --control-plane projects/model-router-lab/outputs/control_plane.json \
  --task-type coding_hard \
  --mode strict
```

8. 模拟控制平面在生产中的效果

```bash
python projects/model-router-lab/routerlab.py simulate-control-plane \
  --control-plane projects/model-router-lab/outputs/control_plane.json \
  --report projects/model-router-lab/outputs/benchmark.json \
  --min-score 7 \
  --max-latency 45 \
  --out projects/model-router-lab/outputs/control_plane_simulation.json
```

9. 自动进化控制平面（推荐）

```bash
python projects/model-router-lab/routerlab.py evolve-control-plane \
  --control-plane projects/model-router-lab/outputs/control_plane.json \
  --report projects/model-router-lab/outputs/benchmark.json \
  --min-score 7 \
  --max-latency 45 \
  --max-iters 3 \
  --out projects/model-router-lab/outputs/control_plane_evolved.json
```

再对进化后的策略做仿真：

```bash
python projects/model-router-lab/routerlab.py simulate-control-plane \
  --control-plane projects/model-router-lab/outputs/control_plane_evolved.json \
  --report projects/model-router-lab/outputs/benchmark.json \
  --min-score 7 \
  --max-latency 45 \
  --out projects/model-router-lab/outputs/control_plane_evolved_simulation.json
```

## 输出说明

- `benchmark.json`: 评分、时延、token 估算、成本估算、相对基线节省
- `policy.json`: 默认模型 + 升级模型 + 按任务类型路由规则 + 重试策略
- `control_plane.json`: 多代理工作流策略（按任务类型拆分 planner/coder/reviewer 阶段）
- `control_plane_simulation.json`: 控制平面仿真结果（预测分数、P95 时延、是否需继续进化）
- `control_plane_evolved.json`: 自动进化后的控制平面策略（系统自动挑选更优 stage 组合）
- `control_plane_evolved_simulation.json`: 进化策略仿真结果（用于上线前 gate）
- `report.html`: 可直接展示的报告页

## 下一步可商业化方向

- 团队版看板：历史趋势、按仓库/团队分层统计
- 规则审批流：低分任务自动升级到更强模型
- GitHub 集成：PR 自动打标并给出模型推荐

## R1 版本亮点

- 质量评估不只靠 LLM 打分，加入 deterministic 校验以降低评估偏差
- 加入 P50/P95 时延指标，方便做 SLA 承诺
- 增加任务类型维度统计，支持 `coding_hard` 与 `summarization` 分开路由

## R2 创新升级（控制平面）

- 从“单步模型路由”升级到“多阶段工作流路由”（multi-agent）
- 每类任务自动生成阶段化执行计划（如 architect -> coder -> reviewer）
- 支持 `standard/quick/strict` 执行模式，按速度和质量切换流程深度
- 支持控制平面仿真，提前评估是否达到创新级上线标准

## R3 创新升级（自动进化）

- 新增 `evolve-control-plane`，自动探索并选择更优 workflow 变体
- 目标函数同时考虑质量阈值、P95 时延和 stage 数量
- 进化日志会记录每轮是否发生结构变化，便于回溯和审计
- 适合作为 nightly job：每天基于最新 benchmark 自动演进策略

## R3.1 稳定性升级（容错评测）

- 当某个模型在评测中返回 API 错误（如 Ollama 500）时，不再中断整轮 benchmark
- 失败会被记录到 `benchmark.meta.model_failures`，其余模型与任务继续执行
- 适用于混合模型环境：单模型不稳定时，整体流程仍可产出策略与仿真结果

## 接入 MiniMax M2.7 做科研

项目已支持混合模型来源：

- 本地 Ollama 模型：直接写模型名（示例：`qwen3:8b`）
- MiniMax 远程模型：使用前缀 `minimax/`（示例：`minimax/MiniMax-M2.7`）

先配置环境变量（PowerShell）：

```powershell
$env:MINIMAX_API_KEY = "你的_API_KEY"
# 可选：如果你有自定义网关
# $env:MINIMAX_API_URL = "https://api.minimaxi.com/v1/text/chatcompletion_v2"
```

检查连通状态：

```bash
python projects/model-router-lab/routerlab.py doctor
```

一键跑科研流水线（推荐）：

```bash
python projects/model-router-lab/routerlab.py run-research-cycle \
  --topic "Discrete diffusion for reasoning" \
  --models minimax/MiniMax-M2.7,qwen3:8b \
  --judge-model qwen3:8b \
  --auto-escalate \
  --max-evolve-iters 3
```

建议在持续迭代时加上 `--run-id`，避免覆盖同名输出并保留历史：

```bash
python projects/model-router-lab/routerlab.py run-research-cycle \
  --topic "Discrete diffusion for reasoning" \
  --arxiv-query "all:discrete diffusion reasoning" \
  --arxiv-max-results 8 \
  --models minimax/MiniMax-M2.7,qwen3:8b \
  --judge-model qwen3:8b \
  --auto-escalate \
  --max-evolve-iters 3 \
  --run-id 20260415_nightly
```

默认情况下，`run-research-cycle` 会先清理同一个 topic 之前生成的旧产物，再写入本次新文件，这样 `outputs/` 不会一直堆积重复结果；同时会保留 `research_history_<topic>.json` 作为摘要历史和 `research_delta_<topic>_<run-id>.json` 作为本次变化记录。

如果你希望保留同一 topic 的历史产物，可以显式加上：

```bash
python projects/model-router-lab/routerlab.py run-research-cycle \
  --topic "Discrete diffusion for reasoning" \
  --arxiv-query "all:discrete diffusion reasoning" \
  --arxiv-max-results 8 \
  --models minimax/MiniMax-M2.7,qwen3:8b \
  --judge-model qwen3:8b \
  --auto-escalate \
  --max-evolve-iters 3 \
  --run-id 20260415_keep \
  --keep-previous-artifacts
```

这个开关适合你想保留每次完整实验痕迹的时候使用；如果只是想让 coworker 输出保持干净，保持默认即可。

产出文件包括：

- 研究任务集（自动生成）
- benchmark / policy / simulation
- control_plane / control_plane_evolved / control_plane simulation
- 可视化 HTML 报告
- `research_history_<topic>.json`：按 topic 记录每次 run 的关键指标
- `research_delta_<topic>_<run-id>.json`：自动对比“本次 vs 上次”分数/时延/判定变化

说明：当 MiniMax 模型临时异常时，流程会记录失败并继续，不会整轮崩溃。

## 开源能力增强：接入 arXiv 文献驱动研究

为了让科研流水线不只基于“主题空想”，项目新增了开源文献检索接入：

- 新命令 `fetch-arxiv`：抓取 arXiv API 结果并落盘
- `make-research-dataset` 支持 `--arxiv-query`：把论文摘要注入任务集
- `run-research-cycle` 支持 `--arxiv-query`：直接跑“文献驱动”的端到端闭环

先单独抓取论文：

```bash
python projects/model-router-lab/routerlab.py fetch-arxiv \
  --query "all:discrete diffusion reasoning" \
  --max-results 8 \
  --out projects/model-router-lab/outputs/arxiv_discrete_diffusion.json
```

再跑文献驱动的科研循环：

```bash
python projects/model-router-lab/routerlab.py run-research-cycle \
  --topic "Discrete diffusion for reasoning" \
  --arxiv-query "all:discrete diffusion reasoning" \
  --arxiv-max-results 8 \
  --models minimax/MiniMax-M2.7,qwen3:8b \
  --judge-model qwen3:8b \
  --auto-escalate \
  --max-evolve-iters 3
```

这样你的 cowork agent 会先消化最新论文摘要，再输出实验假设、ablation 计划与代码 scaffold，最后进入策略优化与自进化。

## 数字人（Digital Person）MVP

你可以基于现有模型路由能力，快速创建一个有稳定人设和记忆的“数字人同事”。

1. 创建数字人人设：

```bash
python projects/model-router-lab/routerlab.py create-digital-person \
  --name "Ava Researcher" \
  --role "AI research coworker" \
  --tone "clear, practical, and proactive" \
  --domain "applied AI research and engineering" \
  --model qwen3:8b
```

2. 与数字人对话（自动保存记忆）：

```bash
python projects/model-router-lab/routerlab.py chat-digital-person \
  --profile projects/model-router-lab/digital_people/ava-researcher.json \
  --message "请给我一个离散扩散推理方向的两周实验计划" \
  --keep-turns 6
```

3. 使用 MiniMax 作为数字人大脑：

```bash
python projects/model-router-lab/routerlab.py chat-digital-person \
  --profile projects/model-router-lab/digital_people/ava-researcher.json \
  --model minimax/MiniMax-M2.7 \
  --message "根据今天的实验结果，建议下一轮 ablation" \
  --keep-turns 8
```

说明：

- 人设文件默认保存在 `projects/model-router-lab/digital_people/`
- 对话记忆默认保存在 `projects/model-router-lab/outputs/digital_person_memory_<id>.json`
- 每次对话会自动读取最近 `--keep-turns` 条历史，让数字人持续保持上下文一致性
