# 本地 Claude 工作流说明

## 1) 先解释你说的 leaking Claude Code

你现在这个仓库不是泄漏出来的私有源码包，而是公开可获取的项目代码镜像。

它包含的是：
- Claude Code 客户端相关仓库内容
- 插件、命令、示例、工作流脚本

它不包含：
- Claude 模型权重
- Anthropic 私有推理后端

所以你能做的是：
- 在本地搭一个高可用的 AI 开发流程
- 用本地模型承担重复任务
- 把你的流程和脚本版本化到你自己的 GitHub 仓库

## 2) 当前已经为你搭好的本地架构

- Claude Code 作为主代理
- Ollama 在本机提供本地模型 API
- MCP 工具 ask_local_qwen 作为分流桥
- 默认模型：qwen3:8b
- 大模型可选：qwen3:32b（需要更高内存/显存）
- 模型存储目录：E:/ollama/models

相关文件：
- .mcp.json
- local_mcp/ollama_mcp.py
- local_mcp/local_ollama_brain.py
- scripts/start_local_claude.ps1
- scripts/start_local_claude.sh
- scripts/start_local_only.ps1
- scripts/start_local_only.sh

## 3) 一键启动方式

PowerShell（推荐，Windows 最稳）：

1. 打开 PowerShell
2. 进入仓库目录
3. 执行：

```powershell
./scripts/start_local_claude.ps1
```

可选参数：

```powershell
./scripts/start_local_claude.ps1 -Model qwen3:8b -ModelStore E:\ollama\models
```

Bash（你偏好的本地 bash）：

```bash
bash scripts/start_local_claude.sh --model qwen3:8b
```

先做自检（不进入 Claude 交互）：

```bash
bash scripts/start_local_claude.sh --model qwen3:8b --skip-pull --no-launch
```

## 4) 在 Claude 里怎么让它自动分流到本地模型

你可以在会话开头给 Claude 这段规则：

```text
你负责架构设计、复杂调试、关键代码修改。
所有批量翻译、批量注释、模板化改写、长文本整理，优先调用 ask_local_qwen。
调用后返回精简结果，并标注需要我人工复核的点。
```

## 5) 你现在可以做什么

- 让本地 qwen3:32b 扛大量重复处理，节省云端 token
- 保留 Claude 做高质量决策和复杂推理
- 把这套流程持续提交到你的仓库，形成你自己的 AI 工具链

## 6) 常见问题

Q: 为什么不是完全离线 Claude？
A: Claude 主体还是云端服务；本地模型是通过 MCP 做任务分流，不是替代 Claude 本身。

Q: 我想让 Claude Code 当主脑，但不登录可以吗？
A: 不可以。Claude Code 作为主脑时，调用的是云端 Claude 服务，必须登录鉴权。你可以把它理解为：
- Claude Code 主脑 = 云端 Claude（需要登录）
- Ollama 本地子模型 = 本地执行器（不需要 Claude 登录）

## 7) 完全不登录 Claude 的纯本地模式（与现有模式并存）

如果你不要登录 Claude，可以直接跑本地 Ollama 主代理（REPL）：

PowerShell:

```powershell
./scripts/start_local_only.ps1 -Model qwen3:8b
```

Bash:

```bash
bash scripts/start_local_only.sh qwen3:8b
```

纯本地模式特点：
- 全程本机，不触发 Claude 登录
- 能做本地问答、代码解释、草稿生成
- 但不等价于 Claude 主脑能力（推理/工具生态会弱一些）

Q: 能换别的本地模型吗？
A: 可以，启动脚本和 MCP 工具都支持传 model 字符串，改成你 ollama list 里的模型名即可。
