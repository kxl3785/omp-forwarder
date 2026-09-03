# omp-forwarder — working notes

A fixed local port in front of Unsloth Studio's `llama-server`. Read
`README.md` first for what it does and why. This file is the things that will
waste your time if you do not know them.

## Layout

```
src/omp_forwarder/
  forwarder.py   the relay, port discovery, CLI. Owns module-level state.
  stats.py       /__stats dashboard: the HTML page and the JSON snapshot.
  usage.py       /__usage page. Polls /__stats.json; no endpoint of its own.
  tray.py        Windows tray icon (win32gui). Imported lazily, only for --tray.
  make_icon.py   generates assets/omp-forwarder.ico. Pure stdlib, no Pillow.
assets/          the .ico, two README screenshots, and two hand-written SVGs
                 (architecture, performance). Regenerate the screenshots with
                 headless Chrome --screenshot against a running forwarder.
                 SVGs are XML: use numeric entities or literal characters,
                 never &mdash; and friends, or GitHub refuses to render them.
run_forwarder.bat  pythonw launcher, puts src/ on PYTHONPATH so a clone works
tests/           unittest suite; see Testing below
bench/           Studio-vs-forwarder measurements behind the README numbers
```

`bench/` needs Studio's API key. It comes from `STUDIO_API_KEY_FILE`, set in
`.claude/settings.local.json` (gitignored) so the scripts can run without the
key ever passing through a command line or a transcript. Never cat that file.

Run it without installing:

```bash
PYTHONPATH=src python -m omp_forwarder --port 8891
```

Use a **spare port** while developing. Something is probably using 8890.

## Constraints that are not obvious

**It is a raw TCP relay, and it must stay one.** It reads only each request's
first line, to route `/__stats` locally. It does not parse HTTP framing,
because a relay that did could break SSE streaming, keep-alive, or chunked
bodies. If you are tempted to parse a body, do not.

**Never bind anything but loopback.** The upstream `llama-server` has no API
key, so this forwarder hands out unauthenticated model access. There is
deliberately no `--host`.

**`sys.modules[__name__]`, never `import omp_forwarder`.** Under some entry
points, importing this package by name while it is already running builds a
*second* module object with its own `_upstream`. Symptom: the dashboard reports
`upstream: null` while the relay is happily serving. Cost an hour.

**Only the FIRST request on a connection is routed.** Anything pipelined
after it follows wherever that one went. This bit once: a browser asks for
`/favicon.ico` before the page loads, that got relayed upstream, and the
page's own `/__stats.json` fetch reused the same keep-alive connection and
reached `llama-server` instead — a 404 and a dashboard of zeros on first
load. `/favicon.ico` is now answered locally with `Connection: close`. Any
new local path needs the same treatment.

**The status counter is intentionally naive.** `_note_status` inspects only
reads that *begin* with a status line. It can undercount; it will not
miscount. That is the right trade for a relay.

## llama-server facts, checked against a live build

- **`kv_cache_usage_ratio` does not exist.** Do not add a KV card; it reads 0
  forever. Dump `/metrics` and check before trusting any metric name.
- **`tokens_predicted_total` only moves when a request COMPLETES.** Use
  `n_decode_total` for anything live, or the page reads "idle" through a long
  generation while a request is plainly in flight. It also makes a rate
  bursty: a 1,932-token reply finishing inside one 3 s poll read as 648
  tok/s. The Throughput card therefore sums the per-stream decode rates from
  `/slots` and falls back to this counter only when `/slots` is unavailable.
- **Per-stream data comes from `/slots`, not `/metrics`.** The counter is
  `next_token[0].n_decoded`.
- **`/metrics` has no error counter.** That is the whole reason the forwarder
  counts status codes itself.
- **A full unified KV pool halves decode speed for everyone.** Studio uses
  `--kv-unified`; a pass attends over the whole occupied pool. Measured
  2026-09-03: 64 tok/s with ~100k tokens of idle agent context in the pool,
  131-135 tok/s with the slots erased, same request. Before you blame the
  model, the proxy, or the GPU, check what the idle slots hold
  (`/slots` `n_prompt_tokens`). `bench/kv_pool.py` is the test.
- **The prompt cache on a hybrid model hits only on exact extension.**
  qwen35 has recurrent SSM layers; their state cannot roll back, and Studio
  passes `--ctx-checkpoints 0`. So a new prompt reuses the slot only if it
  extends the slot's full token sequence, previous reply and
  `reasoning_content` included. An identical re-ask misses. Measured
  2026-09-03: 43 ms vs 6,300 ms on 15k tokens. Any benchmark that fakes the
  assistant turn measures the miss, not the loop. `bench/agent_loop.py` keeps
  the real reply for this reason.
- **Every counter restarts at zero on a model reload**, because a reload is a
  new process. That is why the token total is the forwarder's own tally
  (`_tally_tokens`), not a `/metrics` read. A port change or a counter going
  backwards means a new process, and its counters are added whole.

## Two rate bugs already fixed — do not reintroduce them

**`n_decoded` restarts when a slot takes a new request.** A rate is only valid
while `id_task` is unchanged.

**`decoded == 0` means prefilling, not idle.** A window that straddles the
prefill→decode boundary is mostly prefill, so it reported ~1 tok/s on a
100k-token prompt — true of the window, badly wrong about the stream. Rates are
suppressed for one tick across a phase change. Only decode rates are summed
into the aggregate; adding a prefill rate to a decode rate means nothing.

## Testing

```bash
python -m unittest            # from the repo root; ~20 s, no dependencies
```

`tests/__init__.py` puts `src/` on `sys.path`, so a clone works with nothing
installed. GitHub Actions runs the same command on Ubuntu and Windows, Python
3.10 and 3.13, on every push to main (`.github/workflows/tests.yml`). Nothing in the suite talks to a real llama-server: `tests/helpers.py`
has `FakeUpstream`, a TCP server that speaks just enough HTTP to stand in for
one, and `RelayCase`, which starts the relay on a free port. Discovery tests
replace `tasklist` and `netstat` with recorded output, so they run on any OS.

**The tests share the forwarder's module-level state** and reset it in
`setUp`. They must run sequentially. unittest does; do not add a parallel
runner. `_serve_forever` returns when its listener is closed only so the tests
can stop it.

**`prompt_tokens_total` and `prompt_tokens_cached_total` are disjoint.**
llama-server puts a prompt token in exactly one of them. That is what lets
`/__usage` map them onto a paid API's "input" and "cache read" lines without
estimating. Do not add them together and call it "prompt tokens submitted"
unless you mean the sum of both.

Not covered: the tray, `make_icon`, the two pages' JavaScript, and anything
that needs a live llama-server (real `/metrics` names, real `/slots` shapes).
For those, verify by hand and say what you actually ran:

```bash
PYTHONPATH=src python -m omp_forwarder --port 8891     # starts, discovers
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8891/__stats
curl -s http://127.0.0.1:8891/__stats.json | python -m json.tool | head -30
```

Then send a real completion through it and confirm the dashboard's request and
2xx counters move. **Drive two concurrent requests** whenever you touch the
per-stream table — a single stream hides every bug in it.

For the icon, `python -m omp_forwarder.make_icon` prints an ASCII preview.
Check 16px legibility there rather than opening the file.

## Style

Match the surrounding code. Comments explain *why*, especially where the code
looks wrong but is not — most comments here exist because something failed
once. Keep them.

Do not add dependencies. The forwarder has none, and `pywin32` is optional and
lazily imported so `--tray` degrades to headless rather than failing.
