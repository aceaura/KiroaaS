#!/usr/bin/env python3
"""GPT-5.6-luna 双协议工具调用覆盖测试（gw2）。

Anthropic /v1/messages        -> Claude Code 兼容
OpenAI   /v1/chat/completions -> Codex 兼容
零依赖（urllib）。
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

API_KEY = get_key()

RESULTS = []


def record(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""), flush=True)


def _post(url, body, headers, stream=False):
    headers = {**headers, "Content-Type": "application/json"}
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers=headers, method="POST")
    try:
        resp = urllib.request.urlopen(req, timeout=300)
        return resp.status, resp if stream else resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


# ---------- 工具定义（Claude Code 风格） ----------
ANTHROPIC_TOOLS = [
    {"name": "Read", "description": "Read a file from local disk.",
     "input_schema": {"type": "object",
                      "properties": {"file_path": {"type": "string", "description": "absolute path"},
                                     "offset": {"type": "integer"}, "limit": {"type": "integer"}},
                      "required": ["file_path"]}},
    {"name": "Bash", "description": "Run a shell command.",
     "input_schema": {"type": "object",
                      "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}},
                      "required": ["command"]}},
    {"name": "Edit", "description": "Replace exact text in a file.",
     "input_schema": {"type": "object",
                      "properties": {"file_path": {"type": "string"}, "old_string": {"type": "string"},
                                     "new_string": {"type": "string"}},
                      "required": ["file_path", "old_string", "new_string"]}},
    {"name": "Deploy", "description": "Deploy service with config.",
     "input_schema": {"type": "object",
                      "properties": {"service": {"type": "string"},
                                     "config": {"type": "object",
                                                "properties": {"replicas": {"type": "integer"},
                                                               "env": {"type": "object"},
                                                               "tags": {"type": "array", "items": {"type": "string"}}}},
                                     "regions": {"type": "array", "items": {"type": "string"}}},
                      "required": ["service", "config"]}},
]

OPENAI_TOOLS = [
    {"type": "function", "function": {
        "name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
    for t in ANTHROPIC_TOOLS
]

ANTHROPIC_HEADERS = {"x-api-key": API_KEY, "anthropic-version": "2023-06-01"}
OPENAI_HEADERS = {"Authorization": f"Bearer {API_KEY}"}


def anthropic_body(**kw):
    body = {"model": MODEL, "max_tokens": kw.pop("max_tokens", 1024),
            "output_config": {"effort": "low"}, "tools": ANTHROPIC_TOOLS}
    body.update(kw)
    return body


def openai_body(**kw):
    body = {"model": MODEL, "max_tokens": kw.pop("max_tokens", 1024),
            "reasoning_effort": "low", "tools": OPENAI_TOOLS}
    body.update(kw)
    return body


# ---------- SSE 解析 ----------
def anthropic_stream(body):
    """返回 (blocks, status, delta_count_or_errtext)"""
    body = dict(body, stream=True)
    status, r = _post(f"{BASE}/v1/messages", body, ANTHROPIC_HEADERS, stream=True)
    if status != 200:
        return None, status, r[:300]
    blocks = {}
    delta_count = 0
    for rawline in iter(r.readline, b""):
        line = rawline.decode().strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        ev = json.loads(data)
        t = ev.get("type")
        if t == "content_block_start":
            cb = ev.get("content_block", {})
            blocks[ev["index"]] = {"type": cb.get("type"), "id": cb.get("id"),
                                   "name": cb.get("name"), "json": "", "text": ""}
        elif t == "content_block_delta":
            d = ev.get("delta", {})
            b = blocks.get(ev["index"])
            if b is None:
                continue
            if d.get("type") == "input_json_delta":
                b["json"] += d.get("partial_json", "")
                delta_count += 1
            elif d.get("type") == "text_delta":
                b["text"] += d.get("text", "")
    return blocks, 200, delta_count


def openai_stream(body):
    body = dict(body, stream=True)
    status, r = _post(f"{BASE}/v1/chat/completions", body, OPENAI_HEADERS, stream=True)
    if status != 200:
        return None, status, r[:300]
    calls = {}
    for rawline in iter(r.readline, b""):
        line = rawline.decode().strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        ev = json.loads(data)
        ch = (ev.get("choices") or [{}])[0]
        delta = ch.get("delta") or {}
        for tc in delta.get("tool_calls") or []:
            idx = tc.get("index", 0)
            c = calls.setdefault(idx, {"id": None, "name": None, "args": ""})
            if tc.get("id"):
                c["id"] = tc["id"]
            fn = tc.get("function") or {}
            if fn.get("name"):
                c["name"] = fn["name"]
            if fn.get("arguments"):
                c["args"] += fn["arguments"]
    return calls, 200, None


# ---------- Anthropic 用例 ----------
def a1_nonstream_single():
    body = anthropic_body(
        messages=[{"role": "user", "content": "Use the Read tool to read /etc/hosts."}],
        tool_choice={"type": "tool", "name": "Read"})
    status, text = _post(f"{BASE}/v1/messages", body, ANTHROPIC_HEADERS)
    if status != 200:
        return record("A1 非流式单工具", False, f"HTTP {status}: {text[:200]}")
    d = json.loads(text)
    tu = [b for b in d.get("content", []) if b.get("type") == "tool_use"]
    ok = len(tu) == 1 and tu[0]["name"] == "Read" and isinstance(tu[0]["input"], dict) \
        and tu[0]["input"].get("file_path") and tu[0].get("id")
    record("A1 非流式单工具", bool(ok), json.dumps(tu, ensure_ascii=False)[:200])


def a2_stream_single():
    body = anthropic_body(
        messages=[{"role": "user", "content": "Use the Read tool to read /etc/hosts."}],
        tool_choice={"type": "tool", "name": "Read"})
    blocks, status, extra = anthropic_stream(body)
    if status != 200:
        return record("A2 流式单工具", False, f"HTTP {status}: {extra}")
    tu = [b for b in blocks.values() if b["type"] == "tool_use"]
    detail = f"deltas={extra}"
    if len(tu) != 1 or tu[0]["name"] != "Read" or not tu[0]["id"]:
        return record("A2 流式单工具", False, f"blocks={blocks} {detail}")
    try:
        args = json.loads(tu[0]["json"]) if tu[0]["json"] else {}
    except json.JSONDecodeError as e:
        return record("A2 流式单工具", False, f"碎片重组非法JSON: {e}; raw={tu[0]['json'][:120]!r} {detail}")
    ok = isinstance(args, dict) and args.get("file_path")
    record("A2 流式单工具", bool(ok), f"args={args} {detail}")


def a3_stream_multi():
    body = anthropic_body(max_tokens=2000, messages=[{"role": "user", "content":
        "Call exactly two tools in one response with no text: "
        "Bash with command 'echo one' and Read with file_path '/etc/hosts'."}])
    # 模型单回合是否发两个调用有非确定性（原始流取证确认网关未丢块），最多重试 3 次
    last_detail = ""
    for attempt in range(3):
        blocks, status, extra = anthropic_stream(body)
        if status != 200:
            return record("A3 流式双工具", False, f"HTTP {status}: {extra}")
        tu = [b for b in blocks.values() if b["type"] == "tool_use"]
        last_detail = f"count={len(tu)} deltas={extra} attempt={attempt + 1}"
        if len(tu) < 2:
            continue
        for b in tu:
            try:
                args = json.loads(b["json"]) if b["json"] else {}
            except json.JSONDecodeError as e:
                return record("A3 流式双工具", False, f"{b['name']} 碎片重组非法: {e}; raw={b['json'][:120]!r}")
            if not isinstance(args, dict) or not args:
                return record("A3 流式双工具", False, f"{b['name']} 参数为空: {b['json']!r}")
        return record("A3 流式双工具", True, f"{last_detail} names={[b['name'] for b in tu]}")
    record("A3 流式双工具", False, last_detail)


def a4_roundtrip():
    msgs = [
        {"role": "user", "content": "Read /etc/hosts"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "Reading the file."},
            {"type": "tool_use", "id": "toolu_cov01", "name": "Read",
             "input": {"file_path": "/etc/hosts"}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_cov01",
             "content": "127.0.0.1 localhost\n::1 localhost"}]},
        {"role": "user", "content": "What does the file contain? Answer in one short sentence, no tools."},
    ]
    body = anthropic_body(messages=msgs, max_tokens=300)
    status, text = _post(f"{BASE}/v1/messages", body, ANTHROPIC_HEADERS)
    if status != 200:
        return record("A4 tool_result 回环", False, f"HTTP {status}: {text[:200]}")
    d = json.loads(text)
    txt = "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text")
    record("A4 tool_result 回环", bool(txt.strip()), txt[:120])


def a5_string_input_regression():
    """原始 422 故障载荷：历史 tool_use 的 input 是字符串 ':' """
    msgs = [
        {"role": "user", "content": "Read /etc/hosts"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "toolu_cov02", "name": "Read", "input": ":"}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_cov02", "content": "ok"}]},
        {"role": "user", "content": "Now read /etc/hostname instead."},
    ]
    body = anthropic_body(messages=msgs, tool_choice={"type": "tool", "name": "Read"})
    status, text = _post(f"{BASE}/v1/messages", body, ANTHROPIC_HEADERS)
    if status != 200:
        return record("A5 字符串input回归", False, f"HTTP {status}: {text[:200]}")
    tu = [b for b in json.loads(text).get("content", []) if b.get("type") == "tool_use"]
    ok = bool(tu) and isinstance(tu[0]["input"], dict) and tu[0]["input"].get("file_path")
    record("A5 字符串input回归", ok, json.dumps(tu, ensure_ascii=False)[:150])


def a6_complex_schema():
    body = anthropic_body(max_tokens=1500, messages=[{"role": "user", "content":
        "Call the Deploy tool for service 'web-api' with config: replicas 3, "
        "env {\"LOG_LEVEL\": \"debug\", \"REGION\": \"eu\"}, tags ['v2','canary'], "
        "regions ['eu-west-1','us-east-1']. No text."}],
        tool_choice={"type": "tool", "name": "Deploy"})
    status, text = _post(f"{BASE}/v1/messages", body, ANTHROPIC_HEADERS)
    if status != 200:
        return record("A6 复杂嵌套参数", False, f"HTTP {status}: {text[:200]}")
    tu = [b for b in json.loads(text).get("content", []) if b.get("type") == "tool_use"]
    if not tu:
        return record("A6 复杂嵌套参数", False, "无 tool_use")
    inp = tu[0]["input"]
    ok = isinstance(inp.get("config"), dict) and isinstance(inp.get("config", {}).get("env"), dict) \
        and isinstance(inp.get("regions") or inp.get("config", {}).get("tags"), list)
    record("A6 复杂嵌套参数", bool(ok), json.dumps(inp, ensure_ascii=False)[:220])


def a7_stream_complex():
    body = anthropic_body(max_tokens=1500, messages=[{"role": "user", "content":
        "Call Deploy for service 'batch' with config replicas 2, env {\"MODE\":\"night\"}, "
        "tags ['x'], regions ['ap-1']. No text."}],
        tool_choice={"type": "tool", "name": "Deploy"})
    blocks, status, extra = anthropic_stream(body)
    if status != 200:
        return record("A7 流式复杂参数", False, f"HTTP {status}: {extra}")
    tu = [b for b in blocks.values() if b["type"] == "tool_use"]
    if not tu:
        return record("A7 流式复杂参数", False, "无 tool_use 块")
    try:
        args = json.loads(tu[0]["json"])
    except json.JSONDecodeError as e:
        return record("A7 流式复杂参数", False, f"重组非法: {e}; raw={tu[0]['json'][:150]!r}")
    ok = isinstance(args.get("config"), dict)
    record("A7 流式复杂参数", bool(ok), f"args={json.dumps(args, ensure_ascii=False)[:180]} deltas={extra}")


def a8_long_args_stream():
    long_text = "The quick brown fox jumps over the lazy dog. " * 60  # ~2.7KB
    body = anthropic_body(max_tokens=4096, messages=[{"role": "user", "content":
        f"Call Bash with this exact command: echo '{long_text}' | wc -c . "
        "Put the full command string in the command parameter. No text."}],
        tool_choice={"type": "tool", "name": "Bash"})
    blocks, status, extra = anthropic_stream(body)
    if status != 200:
        return record("A8 超长参数流式", False, f"HTTP {status}: {extra}")
    tu = [b for b in blocks.values() if b["type"] == "tool_use"]
    if not tu:
        return record("A8 超长参数流式", False, "无 tool_use 块")
    try:
        args = json.loads(tu[0]["json"])
    except json.JSONDecodeError as e:
        return record("A8 超长参数流式", False,
                      f"重组非法(碎片={extra}): {e}; len={len(tu[0]['json'])} head={tu[0]['json'][:120]!r} tail={tu[0]['json'][-120:]!r}")
    ok = isinstance(args.get("command"), str) and len(args["command"]) > 500
    record("A8 超长参数流式", bool(ok), f"command_len={len(args.get('command', ''))} deltas={extra}")


def a9_count_tokens():
    body = {"model": MODEL, "tools": ANTHROPIC_TOOLS,
            "messages": [{"role": "user", "content": "Use Read on /etc/hosts"}]}
    status, text = _post(f"{BASE}/v1/messages/count_tokens", body, ANTHROPIC_HEADERS)
    if status != 200:
        return record("A9 count_tokens带tools", False, f"HTTP {status}: {text[:200]}")
    d = json.loads(text)
    ok = isinstance(d.get("input_tokens"), int) and d["input_tokens"] > 0
    record("A9 count_tokens带tools", ok, str(d)[:120])


# ---------- OpenAI 用例 ----------
def o1_nonstream_single():
    body = openai_body(
        messages=[{"role": "user", "content": "Use the Read tool to read /etc/hosts."}],
        tool_choice={"type": "function", "function": {"name": "Read"}})
    status, text = _post(f"{BASE}/v1/chat/completions", body, OPENAI_HEADERS)
    if status != 200:
        return record("O1 非流式单工具", False, f"HTTP {status}: {text[:200]}")
    msg = json.loads(text)["choices"][0]["message"]
    tcs = msg.get("tool_calls") or []
    if len(tcs) != 1:
        return record("O1 非流式单工具", False, f"tool_calls={tcs}")
    tc = tcs[0]
    try:
        args = json.loads(tc["function"]["arguments"])
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        return record("O1 非流式单工具", False, f"arguments非法: {e}; raw={tc['function'].get('arguments')!r}")
    ok = tc["function"]["name"] == "Read" and tc.get("id") and tc.get("type") == "function" \
        and args.get("file_path")
    record("O1 非流式单工具", bool(ok), json.dumps(tc, ensure_ascii=False)[:200])


def o2_stream_single():
    body = openai_body(
        messages=[{"role": "user", "content": "Use the Read tool to read /etc/hosts."}],
        tool_choice={"type": "function", "function": {"name": "Read"}})
    calls, status, extra = openai_stream(body)
    if status != 200:
        return record("O2 流式单工具", False, f"HTTP {status}: {extra}")
    if not calls:
        return record("O2 流式单工具", False, "无 tool_calls delta")
    c = list(calls.values())[0]
    try:
        args = json.loads(c["args"]) if c["args"] else {}
    except json.JSONDecodeError as e:
        return record("O2 流式单工具", False, f"delta重组非法: {e}; raw={c['args'][:120]!r}")
    ok = c["name"] == "Read" and c["id"] and args.get("file_path")
    record("O2 流式单工具", bool(ok), f"args={args}")


def o3_stream_multi():
    body = openai_body(max_tokens=2000, messages=[{"role": "user", "content":
        "Call exactly two tools in one response with no text: "
        "Bash with command 'echo one' and Read with file_path '/etc/hosts'."}],
        parallel_tool_calls=True)
    calls, status, extra = openai_stream(body)
    if status != 200:
        return record("O3 流式双工具", False, f"HTTP {status}: {extra}")
    if len(calls) < 2:
        return record("O3 流式双工具", False, f"count={len(calls)} calls={calls}")
    for idx, c in calls.items():
        try:
            args = json.loads(c["args"]) if c["args"] else {}
        except json.JSONDecodeError as e:
            return record("O3 流式双工具", False, f"#{idx} {c['name']} 重组非法: {e}; raw={c['args'][:120]!r}")
        if not args:
            return record("O3 流式双工具", False, f"#{idx} {c['name']} 参数为空")
    record("O3 流式双工具", True, f"count={len(calls)} names={[c['name'] for c in calls.values()]}")


def o4_roundtrip():
    msgs = [
        {"role": "user", "content": "Read /etc/hosts"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_cov01", "type": "function",
             "function": {"name": "Read", "arguments": "{\"file_path\": \"/etc/hosts\"}"}}]},
        {"role": "tool", "tool_call_id": "call_cov01", "content": "127.0.0.1 localhost"},
        {"role": "user", "content": "What does it contain? One short sentence, no tools."},
    ]
    body = openai_body(messages=msgs, max_tokens=300)
    status, text = _post(f"{BASE}/v1/chat/completions", body, OPENAI_HEADERS)
    if status != 200:
        return record("O4 tool回环", False, f"HTTP {status}: {text[:200]}")
    msg = json.loads(text)["choices"][0]["message"]
    txt = (msg.get("content") or "").strip()
    record("O4 tool回环", bool(txt), txt[:120])


def o5_string_arguments_history():
    """历史里 arguments 是非法 JSON 字符串 ':' —— 与 A5 对称的 OpenAI 侧回归"""
    msgs = [
        {"role": "user", "content": "Read /etc/hosts"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_cov02", "type": "function",
             "function": {"name": "Read", "arguments": ":"}}]},
        {"role": "tool", "tool_call_id": "call_cov02", "content": "ok"},
        {"role": "user", "content": "Now read /etc/hostname instead."},
    ]
    body = openai_body(messages=msgs,
                       tool_choice={"type": "function", "function": {"name": "Read"}})
    status, text = _post(f"{BASE}/v1/chat/completions", body, OPENAI_HEADERS)
    if status != 200:
        return record("O5 非法arguments历史", False, f"HTTP {status}: {text[:200]}")
    tcs = json.loads(text)["choices"][0]["message"].get("tool_calls") or []
    ok = bool(tcs)
    if ok:
        try:
            args = json.loads(tcs[0]["function"]["arguments"])
            ok = bool(args.get("file_path"))
        except json.JSONDecodeError:
            ok = False
    record("O5 非法arguments历史", ok, json.dumps(tcs, ensure_ascii=False)[:150])


def o6_complex_schema():
    body = openai_body(max_tokens=1500, messages=[{"role": "user", "content":
        "Call Deploy for service 'web-api' with config: replicas 3, "
        "env {\"LOG_LEVEL\": \"debug\"}, tags ['v2'], regions ['eu-west-1']. No text."}],
        tool_choice={"type": "function", "function": {"name": "Deploy"}})
    status, text = _post(f"{BASE}/v1/chat/completions", body, OPENAI_HEADERS)
    if status != 200:
        return record("O6 复杂嵌套参数", False, f"HTTP {status}: {text[:200]}")
    tcs = json.loads(text)["choices"][0]["message"].get("tool_calls") or []
    if not tcs:
        return record("O6 复杂嵌套参数", False, "无 tool_calls")
    try:
        args = json.loads(tcs[0]["function"]["arguments"])
    except json.JSONDecodeError as e:
        return record("O6 复杂嵌套参数", False, f"arguments非法: {e}")
    ok = isinstance(args.get("config"), dict)
    record("O6 复杂嵌套参数", bool(ok), json.dumps(args, ensure_ascii=False)[:200])


def o7_long_args_stream():
    long_text = "pack my box with five dozen liquor jugs " * 60
    body = openai_body(max_tokens=4096, messages=[{"role": "user", "content":
        f"Call Bash with this exact command: echo '{long_text}' | wc -c . "
        "Full command in the command parameter. No text."}],
        tool_choice={"type": "function", "function": {"name": "Bash"}})
    calls, status, extra = openai_stream(body)
    if status != 200:
        return record("O7 超长参数流式", False, f"HTTP {status}: {extra}")
    if not calls:
        return record("O7 超长参数流式", False, "无 tool_calls delta")
    c = list(calls.values())[0]
    try:
        args = json.loads(c["args"])
    except json.JSONDecodeError as e:
        return record("O7 超长参数流式", False,
                      f"重组非法: {e}; len={len(c['args'])} head={c['args'][:120]!r} tail={c['args'][-120:]!r}")
    ok = isinstance(args.get("command"), str) and len(args["command"]) > 500
    record("O7 超长参数流式", bool(ok), f"command_len={len(args.get('command', ''))}")


def o8_tool_choice_required():
    body = openai_body(messages=[{"role": "user", "content": "Say hi"}], tool_choice="required")
    status, text = _post(f"{BASE}/v1/chat/completions", body, OPENAI_HEADERS)
    if status != 200:
        return record("O8 tool_choice=required", False, f"HTTP {status}: {text[:200]}")
    tcs = json.loads(text)["choices"][0]["message"].get("tool_calls") or []
    ok = bool(tcs)
    detail = ""
    if ok:
        try:
            json.loads(tcs[0]["function"]["arguments"])
        except json.JSONDecodeError as e:
            ok, detail = False, f"arguments非法: {e}"
    record("O8 tool_choice=required", ok, detail or f"name={tcs[0]['function']['name'] if tcs else None}")


CASES = [a1_nonstream_single, a2_stream_single, a3_stream_multi, a4_roundtrip,
         a5_string_input_regression, a6_complex_schema, a7_stream_complex,
         a8_long_args_stream, a9_count_tokens,
         o1_nonstream_single, o2_stream_single, o3_stream_multi, o4_roundtrip,
         o5_string_arguments_history, o6_complex_schema, o7_long_args_stream,
         o8_tool_choice_required]


def main():
    t0 = time.time()
    for case in CASES:
        try:
            case()
        except Exception as e:  # noqa: BLE001
            record(case.__name__, False, f"EXC {type(e).__name__}: {e}")
    fails = [r for r in RESULTS if not r[1]]
    print(f"\n==== {len(RESULTS) - len(fails)}/{len(RESULTS)} 通过，耗时 {time.time() - t0:.0f}s ====")
    if fails:
        print("失败项:", ", ".join(f[0] for f in fails))
        sys.exit(1)


if __name__ == "__main__":
    main()
