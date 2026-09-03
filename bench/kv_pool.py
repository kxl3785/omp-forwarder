"""Does a full unified KV pool slow down every request?

    python bench/kv_pool.py [--forwarder 8890] [--no-restore]

Studio starts llama-server with --kv-unified: one KV pool shared by all
slots. In that mode a decode pass attends over the whole occupied pool, not
only over the tokens of the request being served, so idle slots holding long
cached contexts can slow down an unrelated short request.

This measures that. It times a 400-token generation with the pool as it is,
then saves every idle slot to disk (llama-server's --slot-save-path), erases
it, times the same generation against an empty pool, and restores the slots
so the other client keeps its prompt cache. The upstream port is read from
the forwarder's /__stats.json.

DESTRUCTIVE if a save fails: the slot is erased anyway, and the client that
owned it re-prefills its context on its next request. It refuses to run
while any slot is busy.

The saved slots stay on disk as bench-kv-pool-slot*.bin in the server's
--slot-save-path directory (about 40 KB per token). The server has no delete
action, so remove them by hand afterwards.

Result on 2026-09-03, Qwen3.8-27B NVFP4 on two RTX 5090s, ~100k tokens held
by idle slots: 64 tok/s as-is, 131-135 tok/s empty, same 400-token request."""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

CODE = ("Write a Python module with a class LRUCache supporting get, put, "
        "and eviction, with docstrings and a small test.")
PROSE = "Write a detailed 800-word essay on the history of TCP ports."


def get(url: str):
    return json.load(urllib.request.urlopen(url, timeout=60))


def post(url: str, body: dict | None = None):
    req = urllib.request.Request(
        url, data=json.dumps(body or {}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        return json.load(urllib.request.urlopen(req, timeout=900))
    except urllib.error.HTTPError as exc:
        return {"http_error": exc.code,
                "body": exc.read().decode(errors="replace")[:300]}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--forwarder", type=int, default=8890)
    p.add_argument("--no-restore", action="store_true",
                   help="leave the pool empty afterwards")
    a = p.parse_args()
    fwd = f"http://127.0.0.1:{a.forwarder}"
    upstream = get(f"{fwd}/__stats.json")["upstream"]
    if not upstream:
        print("forwarder has no upstream", file=sys.stderr)
        return 1
    up = f"http://127.0.0.1:{upstream}"

    def slots():
        return {s["id"]: (s.get("n_prompt_tokens", 0),
                          bool(s.get("is_processing")))
                for s in get(f"{up}/slots")}

    def run(label: str, prompt: str) -> None:
        body = {"messages": [{"role": "user", "content": prompt}],
                "max_tokens": 400, "temperature": 0}
        d = post(f"{fwd}/v1/chat/completions", body)
        t = d["timings"]
        passes = t["predicted_n"] - t.get("draft_n_accepted", 0)
        print(f"  {label:6s} {t['predicted_per_second']:6.1f} tok/s"
              f"  passes/s={passes / (t['predicted_ms'] / 1000):5.1f}"
              f"  tok/pass={t['predicted_n'] / passes:.2f}"
              f"  acceptance={100 * t['draft_n_accepted'] / t['draft_n']:.0f}%",
              flush=True)

    h = slots()
    print("slots (last prompt length, busy):", h)
    if any(busy for _, busy in h.values()):
        print("a slot is busy; not touching anything", file=sys.stderr)
        return 1
    print("POOL AS-IS:")
    run("code", CODE)

    saved = []
    for sid, (n, _) in h.items():
        fn = f"bench-kv-pool-slot{sid}.bin"
        t0 = time.time()
        r = post(f"{up}/slots/{sid}?action=save", {"filename": fn})
        dt = time.time() - t0
        if "http_error" in r:
            print(f"save slot {sid}: FAILED {r}", flush=True)
        else:
            print(f"save slot {sid}: n_saved={r.get('n_saved')}"
                  f"  {r.get('n_written', 0) / 1e9:.2f} GB  {dt:.1f}s",
                  flush=True)
            if r.get("n_saved", 0) > 0:
                saved.append((sid, fn))
        r = post(f"{up}/slots/{sid}?action=erase")
        print(f"erase slot {sid}: {r}", flush=True)

    # The benchmark's own slot now holds ~40 prompt + 400 generated tokens.
    # That is the empty-pool condition.
    print("POOL EMPTY:")
    run("code", CODE)
    run("code", CODE)
    run("prose", PROSE)

    if a.no_restore:
        return 0
    for sid, fn in saved:
        t0 = time.time()
        r = post(f"{up}/slots/{sid}?action=restore", {"filename": fn})
        dt = time.time() - t0
        print(f"restore slot {sid}: "
              f"{r if 'http_error' in r else 'n_restored=%s' % r.get('n_restored')}"
              f"  {dt:.1f}s", flush=True)
    print("slots after restore:", slots())
    print("POOL RESTORED:")
    run("code", CODE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
