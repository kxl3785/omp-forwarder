"""Agent-loop shape: a long, mostly cached context plus a short reply, turn
after turn. Studio's proxy vs the forwarder.

    python bench/agent_loop.py [--context-chars 80000] [--turns 6]
                               [--reply 120] [--studio 8888] [--forwarder 8890]

Each turn appends one exchange to a shared history and sends the whole thing
through both paths. The path that goes first alternates, because the second
call of a turn gets a fuller prompt-cache hit than the first. Per path it
reports the median wall time per turn, the median time to first token, and
the decode rate. This is the number an agent user feels: the wall time.

Studio's port needs its API key. Give it as STUDIO_API_KEY, or set
STUDIO_API_KEY_FILE to a file that holds it. The key is read inside this
process and never printed."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request


def api_key() -> str:
    key = os.environ.get("STUDIO_API_KEY", "")
    path = os.environ.get("STUDIO_API_KEY_FILE", "")
    if not key and path:
        with open(os.path.expanduser(path), encoding="utf-8") as fh:
            key = fh.read().strip()
    return key


def context(chars: int) -> str:
    """Deterministic pseudo-code, so both paths see identical bytes and the
    prompt cache treats them alike."""
    lines = []
    i = 0
    while sum(len(l) + 1 for l in lines) < chars:
        lines.append(f"def handler_{i}(request, ctx):\n"
                     f"    # route {i}: validate, look up record {i * 7 % 1000}, "
                     f"then render\n"
                     f"    rec = ctx.store.get({i * 7 % 1000})\n"
                     f"    if rec is None:\n"
                     f"        return ctx.error(404, 'record {i} missing')\n"
                     f"    return ctx.render('page_{i}.html', rec=rec)\n")
        i += 1
    return "\n".join(lines)


def call(port: int, key: str, messages: list, max_tokens: int):
    """(wall s, time to first token s, tokens streamed) for one turn."""
    body = json.dumps({"messages": messages, "max_tokens": max_tokens,
                       "temperature": 0, "stream": True}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"})
    t0 = time.perf_counter()
    first = None
    n = 0
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            try:
                d = json.loads(payload)
            except ValueError:
                continue
            delta = (d.get("choices") or [{}])[0].get("delta") or {}
            if delta.get("content") or delta.get("reasoning_content"):
                n += 1
                if first is None:
                    first = time.perf_counter() - t0
    return time.perf_counter() - t0, first or float("nan"), n


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--studio", type=int, default=8888)
    p.add_argument("--forwarder", type=int, default=8890)
    p.add_argument("--context-chars", type=int, default=80000,
                   help="size of the fixed system context (~4 chars/token)")
    p.add_argument("--turns", type=int, default=6)
    p.add_argument("--reply", type=int, default=120, help="max tokens per reply")
    a = p.parse_args()
    key = api_key()
    base = [{"role": "system", "content":
             "You maintain this module. Answer briefly.\n\n" +
             context(a.context_chars)}]
    history: list = []
    res = {a.studio: [], a.forwarder: []}
    # Warm: prefill the shared context once, so neither path pays for it.
    try:
        call(a.forwarder, key, base + [{"role": "user", "content": "Ready?"}], 4)
        call(a.studio, key, base + [{"role": "user", "content": "Ready?"}], 4)
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            print("401 from Studio: set STUDIO_API_KEY or STUDIO_API_KEY_FILE",
                  file=sys.stderr)
            return 1
        raise
    t_loop = time.perf_counter()
    for turn in range(a.turns):
        user = {"role": "user", "content":
                f"Turn {turn}: add one more handler to the module and explain "
                f"it in two sentences."}
        msgs = base + history + [user]
        order = (a.studio, a.forwarder) if turn % 2 == 0 else (a.forwarder, a.studio)
        for port in order:
            res[port].append(call(port, key, msgs, a.reply))
        # Same placeholder reply on both paths keeps the shared prefix identical.
        history += [user, {"role": "assistant",
                           "content": f"Added handler_{turn}. It validates the "
                                      f"request and renders page_{turn}."}]
    total = time.perf_counter() - t_loop
    print(f"context ~{a.context_chars // 4:,} tokens, {a.turns} turns, "
          f"{a.reply}-token replies, both paths: {total:.1f} s total")
    for port, xs in res.items():
        label = "Studio   " if port == a.studio else "forwarder"
        wall = statistics.median(x[0] for x in xs)
        ttft = statistics.median(x[1] for x in xs)
        tps = statistics.median((x[2] - 1) / (x[0] - x[1]) for x in xs if x[0] > x[1])
        print(f"{label} :{port}  per turn median {wall:5.2f} s"
              f"  first token {ttft*1000:5.0f} ms  decode {tps:5.1f} tok/s"
              f"  turns {', '.join(f'{x[0]:.2f}' for x in xs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
