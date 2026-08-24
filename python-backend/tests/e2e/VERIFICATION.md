# 网关工具调用验证指引（e2e）

本目录收录针对 Kiro Gateway（gw2，默认 `http://localhost:15235`）的三层工具调用验证：
**Anthropic 协议 `/v1/messages` 面向 Claude Code，OpenAI 协议 `/v1/chat/completions` 面向 Codex**，
默认测试模型 `gpt-5.6-luna`。三个脚本均为零依赖（仅标准库 urllib），直接 `python3` 运行，
**不进 pytest 收集范围**（文件名不匹配 `test_*`）。

## 配置（环境变量）

| 变量 | 默认 | 说明 |
|---|---|---|
| `PROXY_API_KEY` | 无 | 网关 API key；未设置时自动 `docker exec kiro-gateway-gw2 printenv PROXY_API_KEY` 读取 |
| `GW_BASE` | `http://localhost:15235` | 网关地址（测 gw1 改为 `:15234`，注意 gw1 月配额重置日） |
| `GW_MODEL` | `gpt-5.6-luna` | 被测模型 |

## 三层验证

### L1a — 协议覆盖：`tooltest.py`（17 用例，约 220s）

双协议基础矩阵：非流式/流式单工具、流式双工具、tool_result 回环、字符串 input 回归、
复杂嵌套参数、超长参数截断修复、`tool_choice=required`、count_tokens。

```bash
python3 tests/e2e/tooltest.py
```

通过标准：末行 `==== 17/17 通过 ====`，退出码 0。

### L1b — 全工具覆盖：`tooltest_full.py`（82 用例，约 200s）

对 **Claude Code 内建 30 工具**（Read/Write/Edit/Bash/Glob/Grep/WebFetch/WebSearch/TodoWrite/
Agent/TaskOutput/TaskStop/AskUserQuestion/Skill/SlashCommand/EnterPlanMode/ExitPlanMode/
EnterWorktree/ExitWorktree/CronCreate/CronDelete/CronList/ScheduleWakeup/SendMessage/
SendUserFile/ReportFindings/Workflow/DesignSync/MCP 前缀名）与 **Codex 内建 10 工具**
（shell/exec_command/write_stdin/apply_patch/update_plan/view_image/list_mcp_resources/
read_mcp_resource/read_thread_terminal/MCP 前缀名）逐一做
**tool_choice 强制点名 × 非流式 + 流式**，断言工具名精确匹配、参数为合法 JSON 对象、
含期望关键参数；另含双协议工具结果回环。

```bash
python3 tests/e2e/tooltest_full.py          # 全部
python3 tests/e2e/tooltest_full.py A01      # 单用例冒烟
```

通过标准：`==== 82/82 通过 ====`。基线：a8385a3 起两轮 82/82 且全部首次尝试通过。

### L2 — 真实载荷形态：`tooltest_chain.py`（约 30s）

复刻真实 agent 循环：**一次挂载全部工具、不做 tool_choice 强制**，模型自主选择工具 →
喂 tool_result → 继续下一条指令（Anthropic 6 轮 + Codex 3 轮），末轮验证
**一条消息内并行调用两个工具**。与 L1 的单工具隔离形态互补。

```bash
python3 tests/e2e/tooltest_chain.py
```

通过标准：`==== 全部通过 ====`。

### L3 — 真实客户端端到端

#### Claude Code（真实进程直连 gw2，不改 cc-switch 配置）

```bash
KEY=$(docker exec kiro-gateway-gw2 printenv PROXY_API_KEY)
mkdir -p /tmp/cc-e2e && cd /tmp/cc-e2e
ANTHROPIC_BASE_URL=http://localhost:15235 \
ANTHROPIC_AUTH_TOKEN="$KEY" \
ANTHROPIC_MODEL=gpt-5.6-luna ANTHROPIC_SMALL_FAST_MODEL=gpt-5.6-luna \
claude -p "Read the file /etc/hosts, then write a file named result.txt \
containing only its first line, then run: echo cc-e2e-done. Finally reply DONE." \
  --model gpt-5.6-luna --dangerously-skip-permissions
```

验证点：
1. `result.txt` 内容与 `head -1 /etc/hosts` 一致（证明 Read/Write 参数正确且真实执行）；
2. transcript（`~/.claude/projects/<cwd 编码>/*.jsonl`）含 Read/Write/Bash 三个 tool_use；
3. `docker logs kiro-gateway-gw2 --since 10m | grep "/v1/messages"` 有对应记录。

注意：在已有 Claude Code 会话内嵌套运行时，子进程需
`env -u CLAUDECODE -u CLAUDE_CODE_SESSION_ID -u CLAUDE_CODE_CHILD_SESSION -u CLAUDE_PID -u CLAUDE_CODE_ENTRYPOINT`。

#### Codex（日常链路：cc-switch 翻译 → gw2）

Codex CLI 位于 `/Applications/ChatGPT.app/Contents/Resources/codex`。
**codex-cli ≥0.149 已移除 `wire_api="chat"`，只走 Responses API**；gw2 无 `/v1/responses`，
不可直连。日常路径是 cc-switch（`:15721`）把 Responses 翻译成 chat/completions 再发 gw2，
因此直接验证日常路径即覆盖真实链路：

```bash
cd <任一工作目录>
codex exec -c model="gpt-5.6-luna" -c model_reasoning_effort="low" \
  "Create a file named codex_e2e_result.txt containing: hello from codex. \
Then run: cat codex_e2e_result.txt"
```

验证点：
1. 文件被真实创建且 `cat` 输出正确（shell 工具参数经 gw2 往返无误）；
2. `docker logs kiro-gateway-gw2 --since 10m --timestamps | grep chat/completions`
   出现 `model=gpt-5.6-luna` 的多轮流式记录（证明请求确实落在 gw2）。

## 故障定位

- 某用例 FAIL 时脚本会打印网关响应片段；进一步取证：容器内设 `DEBUG_MODE=all` +
  `with TestClient(main.app) as client:`（必须上下文管理器走 lifespan）重放，
  看 `/app/debug_logs/response_stream_raw.txt` 原始帧。
- 单测回归：主机无 pytest，用运行镜像容器跑：
  `docker run --rm --entrypoint sh -v <repo>/python-backend:/src -w /src kiro-gateway:gw2-<tag> -c "pip install -q pytest && python -m pytest tests -q"`
  （注意会连带收集本目录之外的 `tests/unit`、`tests/integration`；本目录脚本不会被收集）。
- 修复后部署：`docker commit` 运行中容器打 base → `FROM base COPY kiro/ /app/kiro/` build →
  手动 `docker run` 重建（勿用 compose），重建前用 `docker inspect` 读全量 env 落 0600 临时文件。

## 历史基线

| 日期 | 提交 | 结果 |
|---|---|---|
| 2026-08-22 | a8385a3 | L1a 17/17 ×2 轮；L1b 82/82 ×2 轮；L2 双协议全过；L3 cc/codex 真实客户端全过；单测 1900 passed |
