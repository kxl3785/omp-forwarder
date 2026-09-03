# omp-forwarder — working notes

A fixed local port in front of Unsloth Studio's `llama-server`. Read
`README.md` first for what it does and why. This file is the things that will
waste your time if you do not know them.

## Layout

```
src/omp_forwarder/
  forwarder.py   the relay, port discovery, CLI. Owns module-level state.
  stats.py       /__stats dashboard: the HTML page and the JSON snapshot.
  tray.py        Windows tray icon (win32gui). Imported lazily, only for --tray.
  make_icon.py   generates assets/omp-forwarder.ico. Pure stdlib, no Pillow.
assets/          the .ico, and the README screenshot
run_forwarder.bat  pythonw launcher, puts src/ on PYTHONPATH so a clone works
```

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

**The status counter is intentionally naive.** `_note_status` inspects only
reads that *begin* with a status line. It can undercount; it will not
miscount. That is the right trade for a relay.

## llama-server facts, checked against a live build

- **`kv_cache_usage_ratio` does not exist.** Do not add a KV card; it reads 0
  forever. Dump `/metrics` and check before trusting any metric name.
- **`tokens_predicted_total` only moves when a request COMPLETES.** Use
  `n_decode_total` for anything live, or the page reads "idle" through a long
  generation while a request is plainly in flight.
- **Per-stream data comes from `/slots`, not `/metrics`.** The counter is
  `next_token[0].n_decoded`.
- **`/metrics` has no error counter.** That is the whole reason the forwarder
  counts status codes itself.

## Two rate bugs already fixed — do not reintroduce them

**`n_decoded` restarts when a slot takes a new request.** A rate is only valid
while `id_task` is unchanged.

**`decoded == 0` means prefilling, not idle.** A window that straddles the
prefill→decode boundary is mostly prefill, so it reported ~1 tok/s on a
100k-token prompt — true of the window, badly wrong about the stream. Rates are
suppressed for one tick across a phase change. Only decode rates are summed
into the aggregate; adding a prefill rate to a decode rate means nothing.

## Testing

There is no test suite yet. That is the most obvious gap. Until there is,
verify by hand and say what you actually ran:

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
