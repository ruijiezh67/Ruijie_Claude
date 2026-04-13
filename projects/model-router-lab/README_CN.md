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

## 输出说明

- `benchmark.json`: 评分、时延、token 估算、成本估算、相对基线节省
- `policy.json`: 默认模型 + 升级模型 + 按任务类型路由规则 + 重试策略
- `report.html`: 可直接展示的报告页

## 下一步可商业化方向

- 团队版看板：历史趋势、按仓库/团队分层统计
- 规则审批流：低分任务自动升级到更强模型
- GitHub 集成：PR 自动打标并给出模型推荐

## R1 版本亮点

- 质量评估不只靠 LLM 打分，加入 deterministic 校验以降低评估偏差
- 加入 P50/P95 时延指标，方便做 SLA 承诺
- 增加任务类型维度统计，支持 `coding_hard` 与 `summarization` 分开路由
