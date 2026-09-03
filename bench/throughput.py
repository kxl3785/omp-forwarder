"""Streamed generation speed: Studio's proxy vs the forwarder.

    python bench/throughput.py [--studio 8888] [--forwarder 8890] [-n 3]
                               [--tokens 400]

Studio re-streams every token, so this is where its proxy costs most. tok/s is
decode only: tokens are counted from the first streamed token to the last, so
prompt processing is excluded.

Studio's port needs its API key. Give it as STUDIO_API_KEY, or set
STUDIO_API_KEY_FILE to a file that holds it. The key is read inside this
process and sent only to 127.0.0.1; it is never printed."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request

PROMPT = "Write a detailed 800-word essay on the history of TCP ports."


def api_key() -> str:
    key = os.environ.get("STUDIO_API_KEY", "")
    path = os.environ.get("STUDIO_API_KEY_FILE", "")
    if not key and path:
        with open(os.path.expanduser(path), encoding="utf-8") as fh:
            key = fh.read().strip()
    return key


def one(port: int, key: str, max_tokens: int) -> tuple[float, int, float]:
    """(decode tok/s, tokens streamed, wall seconds) for one request."""
    body = json.dumps({"messages": [{"role": "user", "content": PROMPT}],
                       "max_tokens": max_tokens, "temperature": 0,
                       "stream": True}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"})
    t0 = time.perf_counter()
    first = last = None
    n = 0
    with urllib.request.urlopen(req, timeout=600) as r:
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
            # Reasoning models stream their thinking as reasoning_content.
            if delta.get("content") or delta.get("reasoning_content"):
                n += 1
                last = time.perf_counter()
                if first is None:
                    first = last
    wall = time.perf_counter() - t0
    tps = n / (last - first) if first and last > first else float("nan")
    return tps, n, wall


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--studio", type=int, default=8888)
    p.add_argument("--forwarder", type=int, default=8890)
    p.add_argument("-n", type=int, default=3, help="requests per path")
    p.add_argument("--tokens", type=int, default=400)
    a = p.parse_args()
    key = api_key()
    ports = (a.studio, a.forwarder)
    try:
        for port in ports:
            one(port, key, 16)                    # warm both paths
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            print("401 from Studio: set STUDIO_API_KEY or STUDIO_API_KEY_FILE",
                  file=sys.stderr)
            return 1
        raise
    res: dict[int, list[tuple[float, int, float]]] = {port: [] for port in ports}
    for _ in range(a.n):
        for port in ports:
            res[port].append(one(port, key, a.tokens))
    for port, xs in res.items():
        label = "Studio   " if port == a.studio else "forwarder"
        tps = [x[0] for x in xs]
        print(f"{label} :{port}  median {statistics.median(tps):6.1f} tok/s"
              f"  runs {', '.join(f'{t:.1f}' for t in tps)}"
              f"  tokens {xs[0][1]}"
              f"  wall {statistics.median(x[2] for x in xs):.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
