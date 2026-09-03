"""Round-trip latency of a short request: Studio's proxy vs the forwarder.

    python bench/latency.py [--studio 8888] [--forwarder 8890] [-n 10]

Studio's port needs its API key. Give it as STUDIO_API_KEY, or set
STUDIO_API_KEY_FILE to a file that holds it. The key is read inside this
process and sent only to 127.0.0.1; it is never printed. The forwarder path
gets the same header, which llama-server ignores.

The two paths are interleaved, so drift in machine state hits both equally.
Both are warmed once before timing."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request

BODY = json.dumps({
    "messages": [{"role": "user", "content": "Reply with the single word: pong"}],
    "max_tokens": 8, "temperature": 0}).encode()


def api_key() -> str:
    key = os.environ.get("STUDIO_API_KEY", "")
    path = os.environ.get("STUDIO_API_KEY_FILE", "")
    if not key and path:
        with open(os.path.expanduser(path), encoding="utf-8") as fh:
            key = fh.read().strip()
    return key


def one(port: int, key: str) -> float:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions", data=BODY,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as r:
        r.read()
    return time.perf_counter() - t0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--studio", type=int, default=8888)
    p.add_argument("--forwarder", type=int, default=8890)
    p.add_argument("-n", type=int, default=10, help="requests per path")
    a = p.parse_args()
    key = api_key()
    ports = (a.studio, a.forwarder)
    try:
        for port in ports:
            one(port, key)
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            print("401 from Studio: set STUDIO_API_KEY or STUDIO_API_KEY_FILE",
                  file=sys.stderr)
            return 1
        raise
    res: dict[int, list[float]] = {port: [] for port in ports}
    for _ in range(a.n):
        for port in ports:
            res[port].append(one(port, key))
    for port, xs in res.items():
        label = "Studio   " if port == a.studio else "forwarder"
        print(f"{label} :{port}  median {statistics.median(xs)*1000:5.0f} ms"
              f"  min {min(xs)*1000:5.0f}  max {max(xs)*1000:5.0f}  (n={a.n})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
