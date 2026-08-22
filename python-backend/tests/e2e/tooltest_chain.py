#!/usr/bin/env python3
"""
tooltest_chain.py — L2 真实载荷形态测试（gpt-5.6-luna 经 gw2）

与 tooltest_full.py 的区别：
- 一次挂载**全部**工具（Claude Code 30 个 / Codex 10 个），不做 tool_choice 强制
- 多轮对话：模型自主选择工具 → 喂 tool_result → 继续指令，复刻真实 agent 循环
- 含并行调用轮（一条消息要求两个工具同时调用）
"""
import json
import sys
import time

from tooltest_full import (
    CC_TOOLS, CODEX_TOOLS, to_anthropic_tool, to_openai_tool,
    _post, H_ANTH, H_OAI, check_call, MODEL,
)

ALL_ANTH_TOOLS = [to_anthropic_tool(t) for t in CC_TOOLS]
ALL_OAI_TOOLS = [to_openai_tool(t) for t in CODEX_TOOLS]
MAX_RETRY = 3

# (tag, 期望工具, 期望参数键, 伪造的工具结果, 该轮成功后追加的下一条用户指令)
ANTH_TURNS = [
    ("read", "Read", ["file_path"], "127.0.0.1 localhost",
     "Now run the command: echo hello"),
    ("bash", "Bash", ["command"], "hello",
     "Now write the file /tmp/chain_test.txt with content 'abc'"),
    ("write", "Write", ["file_path", "content"], "File written",
     "Now edit /tmp/chain_test.txt replacing 'abc' with 'def'"),
    ("edit", "Edit", ["file_path", "old_string", "new_string"], "File edited",
     "Now search for the string TODO in the current project"),
    ("grep", "Grep", ["pattern"], "no matches",
     "Finally record this task in the todo list as completed"),
    ("todo", "TodoWrite", ["todos"], "ok", None),
]

OAI_TURNS = [
    ("shell", "shell", ["command"], "hello",
     "Now create the file hello.txt with content 'hello' using apply_patch"),
    ("patch", "apply_patch", ["input"], "ok",
     "Now update the plan: step 'analyze' completed, step 'implement' "
     "in_progress"),
    ("plan", "update_plan", ["plan"], "ok", None),
]


def anthropic_chain():
    """全 30 工具挂载，6 轮链式 + 1 轮并行"""
    messages = [{"role": "user", "content":
                 "You are Claude Code running with tools. Follow my "
                 "instructions step by step. First: read /etc/hosts."}]
    body_base = {
        "model": MODEL, "max_tokens": 1024,
        "system": "You are Claude Code, an AI coding assistant.",
        "tools": ALL_ANTH_TOOLS,
        "output_config": {"effort": "low"},
    }
    for tag, expect_name, expect_keys, fake_result, next_instr in ANTH_TURNS:
        ok, detail = False, ""
        for attempt in range(1, MAX_RETRY + 1):
            status, text = _post("/v1/messages",
                                 {**body_base, "messages": messages}, H_ANTH)
            if status != 200:
                detail = f"HTTP {status}: {text[:200]}"
                continue
            data = json.loads(text)
            hit = next((b for b in data.get("content", [])
                        if b.get("type") == "tool_use"
                        and b.get("name") == expect_name), None)
            if not hit:
                names = [b.get("name") or b.get("type")
                         for b in data.get("content", [])]
                detail = f"未选中 {expect_name}; blocks={names}"
                continue
            ok, detail = check_call(expect_name, expect_keys, hit.get("input"))
            if not ok:
                continue
            # 回环：assistant 原样 + tool_result
            messages.append({"role": "assistant", "content": data["content"]})
            messages.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": hit["id"],
                 "content": fake_result}]})
            break
        if not ok:
            return False, f"{tag}: {detail}"
        if next_instr:
            messages.append({"role": "user", "content": next_instr})
        print(f"  [chain-A:{tag}] {expect_name} PASS (attempt {attempt})",
              flush=True)

    # 并行轮：一条指令要求两个工具同时调用
    messages.append({"role": "user", "content":
                     "In ONE response, do both at the same time: "
                     "read /etc/hosts AND run echo parallel."})
    detail = ""
    for attempt in range(1, MAX_RETRY + 1):
        status, text = _post("/v1/messages",
                             {**body_base, "messages": messages}, H_ANTH)
        if status != 200:
            detail = f"HTTP {status}: {text[:200]}"
            continue
        data = json.loads(text)
        names = {b.get("name") for b in data.get("content", [])
                 if b.get("type") == "tool_use"}
        if {"Read", "Bash"} <= names:
            print(f"  [chain-A:parallel] Read+Bash PASS (attempt {attempt})",
                  flush=True)
            return True, ""
        detail = f"并行块={sorted(names)}"
    return False, f"并行轮未同时返回 Read+Bash: {detail}"


def openai_chain():
    """全 10 工具挂载，3 轮链式 + 1 轮并行"""
    messages = [
        {"role": "system", "content": "You are Codex, a coding agent."},
        {"role": "user", "content": "Run the command: echo hello"}]
    body_base = {"model": MODEL, "tools": ALL_OAI_TOOLS,
                 "reasoning_effort": "low"}
    for tag, expect_name, expect_keys, fake_result, next_instr in OAI_TURNS:
        ok, detail = False, ""
        for attempt in range(1, MAX_RETRY + 1):
            status, text = _post("/v1/chat/completions",
                                 {**body_base, "messages": messages}, H_OAI)
            if status != 200:
                detail = f"HTTP {status}: {text[:200]}"
                continue
            data = json.loads(text)
            msg = data.get("choices", [{}])[0].get("message", {})
            hit = next((c for c in msg.get("tool_calls") or []
                        if c.get("function", {}).get("name") == expect_name),
                       None)
            if not hit:
                names = [c.get("function", {}).get("name")
                         for c in msg.get("tool_calls") or []]
                detail = (f"未选中 {expect_name}; calls={names}, "
                          f"content={str(msg.get('content'))[:100]!r}")
                continue
            raw = hit.get("function", {}).get("arguments", "")
            try:
                args_obj = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                args_obj = None
            ok, detail = check_call(expect_name, expect_keys, args_obj)
            if not ok:
                continue
            messages.append({"role": "assistant",
                             "content": msg.get("content"),
                             "tool_calls": msg["tool_calls"]})
            messages.append({"role": "tool",
                             "tool_call_id": hit.get("id", "call_1"),
                             "content": fake_result})
            break
        if not ok:
            return False, f"{tag}: {detail}"
        if next_instr:
            messages.append({"role": "user", "content": next_instr})
        print(f"  [chain-O:{tag}] {expect_name} PASS (attempt {attempt})",
              flush=True)

    # 并行轮
    messages.append({"role": "user", "content":
                     "In ONE response, do both at the same time: "
                     "run echo parallel AND view the image /tmp/a.png."})
    detail = ""
    for attempt in range(1, MAX_RETRY + 1):
        status, text = _post("/v1/chat/completions",
                             {**body_base, "messages": messages}, H_OAI)
        if status != 200:
            detail = f"HTTP {status}: {text[:200]}"
            continue
        data = json.loads(text)
        msg = data.get("choices", [{}])[0].get("message", {})
        names = {c.get("function", {}).get("name")
                 for c in msg.get("tool_calls") or []}
        if {"shell", "view_image"} <= names:
            print(f"  [chain-O:parallel] shell+view_image PASS "
                  f"(attempt {attempt})", flush=True)
            return True, ""
        detail = f"并行块={sorted(n for n in names if n)}"
    return False, f"并行轮未同时返回 shell+view_image: {detail}"


def main():
    t0 = time.time()
    ok_a, det_a = anthropic_chain()
    print(f"[CHAIN-A] anthropic 全工具链 ... "
          f"{'PASS' if ok_a else 'FAIL: ' + det_a}", flush=True)
    ok_o, det_o = openai_chain()
    print(f"[CHAIN-O] openai 全工具链 ... "
          f"{'PASS' if ok_o else 'FAIL: ' + det_o}", flush=True)
    total_ok = ok_a and ok_o
    print(f"==== {'全部通过' if total_ok else '存在失败'}，"
          f"耗时 {time.time()-t0:.0f}s ====")
    sys.exit(0 if total_ok else 1)


if __name__ == "__main__":
    main()
