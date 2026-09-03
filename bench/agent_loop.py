"""Agent-loop shape: a long, mostly cached context plus a short reply, turn
after turn. Studio's proxy vs the forwarder.

    python bench/agent_loop.py [--context-chars 80000] [--turns 6]
                               [--reply 120] [--studio 8888] [--forwarder 8890]

Each path runs its own loop: a fixed context, then several turns where the
model's real reply, reasoning included, is appended before the next question.
That is the only shape that hits the prompt cache on a hybrid model like
qwen35 with context checkpoints off: the recurrent layers cannot roll back,
so the cache is reused only when the new prompt extends the slot's exact
token sequence. Drop the reasoning, edit history, or re-ask a finished turn
and the whole prompt is prefilled again (measured: 43 ms vs 6,300 ms on a
15k-token context). The paths run one after the other rather than
interleaved, because two requests for the same prefix land on the same slot
and the second one would need a rollback.

Per path it reports the median wall time per turn, the median time to first
token, and the decode rate. Wall time is what an agent user feels.

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
    """(wall s, time to first token s, tokens streamed, reply message)."""
    body = json.dumps({"messages": messages, "max_tokens": max_tokens,
                       "temperature": 0, "stream": True}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"})
    t0 = time.perf_counter()
    first = None
    n = 0
    content, reasoning = [], []
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
            if delta.get("content"):
                content.append(delta["content"])
            if delta.get("reasoning_content"):
                reasoning.append(delta["reasoning_content"])
    # The reply goes back into history verbatim, reasoning included. Without
    # the reasoning the next prompt diverges from the slot's tokens and the
    # cache misses.
    reply = {"role": "assistant", "content": "".join(content)}
    if reasoning:
        reply["reasoning_content"] = "".join(reasoning)
    return time.perf_counter() - t0, first or float("nan"), n, reply


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
    res = {a.studio: [], a.forwarder: []}
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
    for port in (a.studio, a.forwarder):
        history: list = []
        for turn in range(a.turns):
            user = {"role": "user", "content":
                    f"Turn {turn}: add one more handler to the module and "
                    f"explain it in two sentences."}
            wall, ttft, n, reply = call(port, key, base + history + [user],
                                        a.reply)
            res[port].append((wall, ttft, n))
            history += [user, reply]
    total = time.perf_counter() - t_loop
    print(f"context ~{a.context_chars // 4:,} tokens, {a.turns} turns per "
          f"path, {a.reply}-token replies, real replies kept: {total:.1f} s "
          f"total. Turn 0 of each path prefills the context; the rest should "
          f"hit the cache.")
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
