#!/usr/bin/env python3
"""
tooltest_full.py — gpt-5.6-luna 全工具覆盖测试（经 gw2 :15235）

- Anthropic 协议（/v1/messages）= Claude Code 内建工具全集
- OpenAI 协议（/v1/chat/completions）= Codex 内建工具全集
- 每个工具：tool_choice 强制点名 × 非流式 + 流式，各最多重试 3 次
- 断言：调用了指定工具名、参数为合法 JSON 对象、含期望关键参数

用法：
  python3 tooltest_full.py            # 全部
  python3 tooltest_full.py A01        # 只跑单个 case（冒烟）
"""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

BASE = os.environ.get("GW_BASE", "http://localhost:15235")
MODEL = os.environ.get("GW_MODEL", "gpt-5.6-luna")


def get_key():
    k = os.environ.get("PROXY_API_KEY")
    if k:
        return k.strip()
    return subprocess.check_output(
        ["docker", "exec", "kiro-gateway-gw2", "printenv", "PROXY_API_KEY"],
        text=True,
    ).strip()


MAX_RETRY = 3
TIMEOUT = 180


KEY = get_key()
H_ANTH = {"x-api-key": KEY, "anthropic-version": "2023-06-01"}
H_OAI = {"Authorization": f"Bearer {KEY}"}


def _post(path, body, headers):
    """非流式 POST，返回 (status, text)"""
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        BASE + path, data=data, method="POST",
        headers={**headers, "Content-Type": "application/json"},
    )
    try:
        resp = urllib.request.urlopen(req, timeout=TIMEOUT)
        return resp.getcode(), resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except Exception as e:  # 超时/连接错误
        return -1, f"{type(e).__name__}: {e}"


def _post_stream(path, body, headers):
    """流式 POST，返回 (status, resp|err_text)"""
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        BASE + path, data=data, method="POST",
        headers={**headers, "Content-Type": "application/json"},
    )
    try:
        resp = urllib.request.urlopen(req, timeout=TIMEOUT)
        return resp.getcode(), resp
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except Exception as e:
        return -1, f"{type(e).__name__}: {e}"


def anthropic_stream(resp):
    """解析 Anthropic SSE，返回 blocks: [{type,name,id,input_str,text}]"""
    blocks = {}
    order = []
    cur = None
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            ev = json.loads(payload)
        except json.JSONDecodeError:
            continue
        t = ev.get("type")
        if t == "content_block_start":
            idx = ev.get("index", 0)
            cb = ev.get("content_block", {})
            cur = idx
            blocks[idx] = {
                "type": cb.get("type"),
                "name": cb.get("name"),
                "id": cb.get("id"),
                "input_str": "",
                "text": "",
            }
            order.append(idx)
        elif t == "content_block_delta":
            idx = ev.get("index", cur)
            if idx not in blocks:
                blocks[idx] = {"type": None, "name": None, "id": None,
                               "input_str": "", "text": ""}
                order.append(idx)
            d = ev.get("delta", {})
            if d.get("type") == "input_json_delta":
                blocks[idx]["input_str"] += d.get("partial_json", "")
            elif d.get("type") == "text_delta":
                blocks[idx]["text"] += d.get("text", "")
    out = []
    for idx in order:
        b = blocks[idx]
        if b["type"] == "tool_use":
            try:
                b["input"] = json.loads(b["input_str"]) if b["input_str"] else {}
            except json.JSONDecodeError:
                b["input"] = None
        out.append(b)
    return out


def openai_stream(resp):
    """解析 OpenAI SSE，返回 (tool_calls, text, finish_reason)"""
    calls = {}
    text = ""
    finish = None
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            ev = json.loads(payload)
        except json.JSONDecodeError:
            continue
        for ch in ev.get("choices", []):
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
            d = ch.get("delta", {})
            if d.get("content"):
                text += d["content"]
            for tc in d.get("tool_calls") or []:
                i = tc.get("index", 0)
                c = calls.setdefault(i, {"id": "", "name": "", "arguments": ""})
                if tc.get("id"):
                    c["id"] += tc["id"]
                fn = tc.get("function", {})
                if fn.get("name"):
                    c["name"] += fn["name"]
                if fn.get("arguments"):
                    c["arguments"] += fn["arguments"]
    out = []
    for i in sorted(calls):
        c = calls[i]
        try:
            c["args_obj"] = json.loads(c["arguments"]) if c["arguments"] else {}
        except json.JSONDecodeError:
            c["args_obj"] = None
        out.append(c)
    return out, text, finish


# ----------------------------------------------------------------------------
# 工具定义
# ----------------------------------------------------------------------------

def T(name, desc, props, required, expect, hint):
    return {
        "name": name, "desc": desc,
        "schema": {"type": "object", "properties": props, "required": required},
        "expect": expect, "hint": hint,
    }


S = lambda d=None: {"type": "string", **(d or {})}
I = lambda d=None: {"type": "integer", **(d or {})}
B = lambda d=None: {"type": "boolean", **(d or {})}
ARR = lambda items, d=None: {"type": "array", "items": items, **(d or {})}

# ---- Claude Code 内建工具全集（Anthropic 协议）----
CC_TOOLS = [
    T("Read", "Read a file from disk",
      {"file_path": S(), "offset": I(), "limit": I()},
      ["file_path"], ["file_path"], "read the file /etc/hosts"),
    T("Write", "Write a file to disk",
      {"file_path": S(), "content": S()},
      ["file_path", "content"], ["file_path", "content"],
      "write the file /tmp/hello.txt with content 'hello world'"),
    T("Edit", "Replace a string in a file",
      {"file_path": S(), "old_string": S(), "new_string": S(), "replace_all": B()},
      ["file_path", "old_string", "new_string"],
      ["file_path", "old_string", "new_string"],
      "edit /tmp/hello.txt replacing 'hello' with 'hi'"),
    T("NotebookEdit", "Edit a Jupyter notebook cell",
      {"notebook_path": S(), "new_source": S(), "cell_id": S(),
       "cell_type": S({"enum": ["code", "markdown"]}),
       "edit_mode": S({"enum": ["replace", "insert", "delete"]})},
      ["notebook_path", "new_source"], ["notebook_path", "new_source"],
      "replace cell 1 of /tmp/nb.ipynb with the code print('hi')"),
    T("Bash", "Execute a shell command",
      {"command": S(), "description": S(), "timeout": I(), "run_in_background": B()},
      ["command"], ["command"], "run the command: echo hello"),
    T("Glob", "Find files by name pattern",
      {"pattern": S(), "path": S()},
      ["pattern"], ["pattern"], "find all *.py files under the current project"),
    T("Grep", "Search file contents with a regex",
      {"pattern": S(), "path": S(),
       "output_mode": S({"enum": ["content", "files_with_matches", "count"]}),
       "glob": S(), "-i": B(), "head_limit": I()},
      ["pattern"], ["pattern"], "search for the string TODO in the project"),
    T("WebFetch", "Fetch a URL and answer a prompt about it",
      {"url": S(), "prompt": S()},
      ["url", "prompt"], ["url", "prompt"],
      "fetch https://example.com and summarize it"),
    T("WebSearch", "Search the web",
      {"query": S(), "allowed_domains": ARR(S()), "blocked_domains": ARR(S())},
      ["query"], ["query"], "search the web for the current Claude Code version"),
    T("TodoWrite", "Write the session todo list",
      {"todos": ARR({"type": "object", "properties": {
          "content": S(),
          "status": S({"enum": ["pending", "in_progress", "completed"]}),
          "activeForm": S()},
          "required": ["content", "status", "activeForm"]})},
      ["todos"], ["todos"],
      "create a todo list with one task: fix the login bug"),
    T("Agent", "Launch a subagent to handle a task",
      {"description": S(), "prompt": S(), "subagent_type": S(), "model": S(),
       "isolation": S(), "run_in_background": B()},
      ["description", "prompt"], ["description", "prompt"],
      "launch an Explore agent to find where logging is configured"),
    T("TaskOutput", "Read output of a background task",
      {"task_id": S(), "block": B(), "timeout": I()},
      ["task_id"], ["task_id"], "get the output of background task abc123"),
    T("TaskStop", "Stop a background task",
      {"task_id": S()},
      ["task_id"], ["task_id"], "stop the background task abc123"),
    T("AskUserQuestion", "Ask the user a multiple-choice question",
      {"questions": ARR({"type": "object", "properties": {
          "question": S(), "header": S(),
          "options": ARR({"type": "object", "properties": {
              "label": S(), "description": S()},
              "required": ["label", "description"]}),
          "multiSelect": B()},
          "required": ["question", "header", "options", "multiSelect"]})},
      ["questions"], ["questions"],
      "ask the user which database to use: Postgres or SQLite"),
    T("Skill", "Invoke a named skill",
      {"skill": S(), "args": S()},
      ["skill"], ["skill"], "invoke the pdf skill"),
    T("SlashCommand", "Run a slash command",
      {"command": S()},
      ["command"], ["command"], "run the /compact command"),
    T("EnterPlanMode", "Enter read-only plan mode",
      {}, [], [], "enter plan mode"),
    T("ExitPlanMode", "Exit plan mode and present the plan",
      {"allowedPrompts": ARR({"type": "object", "properties": {
          "tool": S(), "prompt": S()}})},
      [], [], "exit plan mode so implementation can start"),
    T("EnterWorktree", "Create and enter a git worktree",
      {"name": S()},
      [], [], "create a worktree named feature-x"),
    T("ExitWorktree", "Exit the current git worktree",
      {"action": S({"enum": ["keep", "remove"]}), "discard_changes": B()},
      ["action"], ["action"], "exit the worktree and keep it"),
    T("CronCreate", "Schedule a future prompt",
      {"cron": S(), "prompt": S(), "recurring": B(), "durable": B()},
      ["cron", "prompt"], ["cron", "prompt"],
      "remind me every hour to check the deploy"),
    T("CronDelete", "Delete a scheduled cron job",
      {"id": S()},
      ["id"], ["id"], "delete the cron job with id abc123"),
    T("CronList", "List scheduled cron jobs",
      {}, [], [], "list all cron jobs"),
    T("ScheduleWakeup", "Schedule a wakeup for a dynamic loop",
      {"delaySeconds": I(), "prompt": S(), "reason": S(), "stop": B()},
      [], ["delaySeconds"],
      "wake up in 1200 seconds to check the loop again"),
    T("SendMessage", "Send a message to another agent",
      {"to": S(), "message": S(), "summary": S()},
      ["to", "message"], ["to", "message"],
      "send a message to the agent named researcher saying: start task 1"),
    T("SendUserFile", "Send a file to the user",
      {"files": ARR(S()), "status": S({"enum": ["normal", "proactive"]}),
       "caption": S(), "display": S({"enum": ["render", "attach"]})},
      ["files", "status"], ["files", "status"],
      "send the file report.md to the user"),
    T("ReportFindings", "Report code review findings",
      {"findings": ARR({"type": "object", "properties": {
          "file": S(), "summary": S(), "failure_scenario": S(),
          "line": I(), "category": S(), "short_summary": S()},
          "required": ["file", "summary", "failure_scenario"]}),
       "level": S({"enum": ["low", "medium", "high", "xhigh", "max"]})},
      ["findings"], ["findings"],
      "report one correctness finding in main.py line 10"),
    T("Workflow", "Run a multi-agent workflow script",
      {"script": S(), "args": {"type": "object"}, "name": S(), "scriptPath": S()},
      ["script"], ["script"],
      "run a trivial workflow whose script is exactly: "
      "export const meta = {name:'t',description:'t'}; return 1"),
    T("DesignSync", "Sync a design system project",
      {"method": S({"enum": ["list_projects", "get_project", "list_files",
                             "get_file", "finalize_plan", "write_files",
                             "delete_files", "register_assets",
                             "unregister_assets", "create_project",
                             "report_validate"]}),
       "projectId": S(), "name": S()},
      ["method"], ["method"], "list my design system projects"),
    T("mcp__Claude_Browser__preview_snapshot", "Accessibility snapshot of the page",
      {"serverId": S()},
      [], [], "take an accessibility snapshot of the browser page"),
]

# ---- Codex 内建工具全集（OpenAI 协议）----
CODEX_TOOLS = [
    T("shell", "Execute a shell command",
      {"command": ARR(S()), "workdir": S(), "timeout_ms": I(), "login": B(),
       "sandbox_permissions": S(), "justification": S()},
      ["command"], ["command"], "run the command echo hello"),
    T("exec_command", "Execute a command in a PTY session",
      {"cmd": S(), "workdir": S(), "yield_time_ms": I(), "max_output_tokens": I()},
      ["cmd"], ["cmd"], "run ls -la"),
    T("write_stdin", "Write characters to a running PTY session",
      {"session_id": I(), "chars": S(), "yield_time_ms": I(),
       "max_output_tokens": I()},
      ["session_id"], ["session_id"], "send the character q to session 12"),
    T("apply_patch", "Apply a patch in the apply_patch DSL format",
      {"input": S()},
      ["input"], ["input"],
      "apply a patch that adds the file hello.txt with content hello"),
    T("update_plan", "Update the task plan",
      {"plan": ARR({"type": "object", "properties": {
          "step": S(),
          "status": S({"enum": ["pending", "in_progress", "completed"]})},
          "required": ["step", "status"]}),
       "explanation": S()},
      ["plan"], ["plan"],
      "set the plan: step 'analyze' in_progress, step 'implement' pending"),
    T("view_image", "View a local image file",
      {"path": S()},
      ["path"], ["path"], "view the image at /tmp/a.png"),
    T("list_mcp_resources", "List resources exposed by MCP servers",
      {"server": S(), "cursor": S()},
      [], [], "list all MCP resources"),
    T("read_mcp_resource", "Read one MCP resource",
      {"server": S(), "uri": S()},
      ["server", "uri"], ["server", "uri"],
      "read the resource file:///tmp/a.txt from the MCP server named fs"),
    T("read_thread_terminal", "Read the shared thread terminal output",
      {}, [], [], "read the current thread terminal output"),
    T("mcp__github__get_issue", "Get a GitHub issue",
      {"owner": S(), "repo": S(), "issue_number": I()},
      ["owner", "repo", "issue_number"], ["owner", "repo"],
      "get issue number 1 from the repo octocat/hello-world"),
]


def to_anthropic_tool(t):
    return {"name": t["name"], "description": t["desc"],
            "input_schema": t["schema"]}


def to_openai_tool(t):
    return {"type": "function", "function": {
        "name": t["name"], "description": t["desc"],
        "parameters": t["schema"]}}


# ----------------------------------------------------------------------------
# 校验
# ----------------------------------------------------------------------------

def check_call(name, expect, args_obj):
    """参数对象校验，返回 (ok, detail)"""
    if args_obj is None:
        return False, "arguments 不是合法 JSON"
    if not isinstance(args_obj, dict):
        return False, f"arguments 不是对象: {type(args_obj).__name__}"
    missing = [k for k in expect if k not in args_obj]
    if missing:
        return False, f"缺参数 {missing}; 实际={list(args_obj.keys())}"
    return True, ""


# ----------------------------------------------------------------------------
# 用例执行
# ----------------------------------------------------------------------------

results = []  # (case_id, ok, attempts, detail)


def run_case(case_id, proto, tool, stream):
    """单用例：强制点名调用一个工具。返回 (ok, attempts, detail)"""
    if proto == "anthropic":
        body = {
            "model": MODEL, "max_tokens": 1024,
            "system": "You are Claude Code, an AI coding assistant.",
            "messages": [{"role": "user", "content":
                          f"Use the {tool['name']} tool right now to "
                          f"{tool['hint']}. Do not answer with plain text."}],
            "tools": [to_anthropic_tool(tool)],
            "tool_choice": {"type": "tool", "name": tool["name"]},
            "output_config": {"effort": "low"},
            "stream": stream,
        }
        path, headers = "/v1/messages", H_ANTH
    else:
        body = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "You are Codex, a coding agent."},
                {"role": "user", "content":
                 f"Use the {tool['name']} tool right now to "
                 f"{tool['hint']}. Do not answer with plain text."}],
            "tools": [to_openai_tool(tool)],
            "tool_choice": {"type": "function",
                            "function": {"name": tool["name"]}},
            "reasoning_effort": "low",
            "stream": stream,
        }
        path, headers = "/v1/chat/completions", H_OAI

    last = ""
    for attempt in range(1, MAX_RETRY + 1):
        if stream:
            status, r = _post_stream(path, body, headers)
            if status != 200:
                last = f"HTTP {status}: {str(r)[:300]}"
                continue
            if proto == "anthropic":
                blocks = anthropic_stream(r)
                hit = next((b for b in blocks
                            if b.get("type") == "tool_use"
                            and b.get("name") == tool["name"]), None)
                if not hit:
                    kinds = [(b.get("type"), b.get("name")) for b in blocks]
                    last = f"流中无目标 tool_use; blocks={kinds}"
                    continue
                ok, last = check_call(tool["name"], tool["expect"],
                                      hit.get("input"))
            else:
                calls, text, finish = openai_stream(r)
                hit = next((c for c in calls
                            if c.get("name") == tool["name"]), None)
                if not hit:
                    last = (f"流中无目标 tool_calls; "
                            f"calls={[c.get('name') for c in calls]}, "
                            f"text={text[:120]!r}")
                    continue
                ok, last = check_call(tool["name"], tool["expect"],
                                      hit.get("args_obj"))
        else:
            status, text = _post(path, body, headers)
            if status != 200:
                last = f"HTTP {status}: {text[:300]}"
                continue
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                last = f"响应非 JSON: {text[:200]}"
                continue
            if proto == "anthropic":
                hit = next((b for b in data.get("content", [])
                            if b.get("type") == "tool_use"
                            and b.get("name") == tool["name"]), None)
                if not hit:
                    kinds = [(b.get("type"), b.get("name"))
                             for b in data.get("content", [])]
                    last = f"响应无目标 tool_use; blocks={kinds}"
                    continue
                ok, last = check_call(tool["name"], tool["expect"],
                                      hit.get("input"))
            else:
                msg = data.get("choices", [{}])[0].get("message", {})
                hit = next((c for c in msg.get("tool_calls") or []
                            if c.get("function", {}).get("name")
                            == tool["name"]), None)
                if not hit:
                    last = (f"响应无目标 tool_calls; "
                            f"content={str(msg.get('content'))[:120]!r}")
                    continue
                raw_args = hit.get("function", {}).get("arguments", "")
                try:
                    args_obj = json.loads(raw_args) if raw_args else {}
                except json.JSONDecodeError:
                    args_obj = None
                ok, last = check_call(tool["name"], tool["expect"], args_obj)
        if ok:
            return True, attempt, ""
    return False, MAX_RETRY, last


def run_roundtrip(case_id, proto):
    """工具结果回环：assistant 工具调用 + 工具结果 → 期待最终文本"""
    if proto == "anthropic":
        body = {
            "model": MODEL, "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "Run echo hello and tell me the output."},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "toolu_rt1",
                     "name": "Bash", "input": {"command": "echo hello"}}]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_rt1",
                     "content": "hello"}]},
            ],
            "tools": [to_anthropic_tool(CC_TOOLS[4])],  # Bash
            "output_config": {"effort": "low"},
        }
        status, text = _post("/v1/messages", body, H_ANTH)
        if status != 200:
            return False, 1, f"HTTP {status}: {text[:300]}"
        data = json.loads(text)
        out = "".join(b.get("text", "") for b in data.get("content", [])
                      if b.get("type") == "text")
        ok = "hello" in out.lower()
        return ok, 1, "" if ok else f"最终文本未提及结果: {out[:150]!r}"
    else:
        body = {
            "model": MODEL,
            "messages": [
                {"role": "user", "content": "Run echo hello and tell me the output."},
                {"role": "assistant", "content": None, "tool_calls": [{
                    "id": "call_rt1", "type": "function",
                    "function": {"name": "shell",
                                 "arguments": json.dumps(
                                     {"command": ["echo", "hello"]})}}]},
                {"role": "tool", "tool_call_id": "call_rt1", "content": "hello"},
            ],
            "tools": [to_openai_tool(CODEX_TOOLS[0])],  # shell
            "reasoning_effort": "low",
        }
        status, text = _post("/v1/chat/completions", body, H_OAI)
        if status != 200:
            return False, 1, f"HTTP {status}: {text[:300]}"
        data = json.loads(text)
        out = data.get("choices", [{}])[0].get("message", {}).get("content") or ""
        ok = "hello" in out.lower()
        return ok, 1, "" if ok else f"最终文本未提及结果: {out[:150]!r}"


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    cases = []
    n = 0
    for t in CC_TOOLS:
        n += 1
        cases.append((f"A{n:02d}", "anthropic", t, False))
        cases.append((f"A{n:02d}s", "anthropic", t, True))
    m = 0
    for t in CODEX_TOOLS:
        m += 1
        cases.append((f"O{m:02d}", "openai", t, False))
        cases.append((f"O{m:02d}s", "openai", t, True))
    cases.append(("RT-A", "roundtrip-anthropic", None, False))
    cases.append(("RT-O", "roundtrip-openai", None, False))

    passed = failed = 0
    t0 = time.time()
    for case_id, proto, tool, stream in cases:
        if only and case_id != only:
            continue
        ts = time.time()
        if proto.startswith("roundtrip"):
            ok, att, detail = run_roundtrip(case_id, proto.split("-")[1])
            label = proto
        else:
            ok, att, detail = run_case(case_id, proto, tool, stream)
            label = f"{proto}:{tool['name']}{'[stream]' if stream else ''}"
        dt = time.time() - ts
        mark = "PASS" if ok else "FAIL"
        print(f"[{case_id}] {label} ... {mark} "
              f"(attempts={att}, {dt:.1f}s) {detail}", flush=True)
        if ok:
            passed += 1
        else:
            failed += 1
        results.append((case_id, ok, att, detail))

    total = passed + failed
    print(f"==== {passed}/{total} 通过，耗时 {time.time()-t0:.0f}s ====")
    if failed:
        print("失败用例：")
        for cid, ok, att, detail in results:
            if not ok:
                print(f"  {cid}: {detail}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
