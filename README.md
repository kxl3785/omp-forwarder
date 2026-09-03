# OMP Forwarder

[![tests](https://github.com/kxl3785/omp-forwarder/actions/workflows/tests.yml/badge.svg)](https://github.com/kxl3785/omp-forwarder/actions/workflows/tests.yml)

A fixed local port in front of Unsloth Studio's `llama-server`, so your client
talks to the model directly instead of through Studio's proxy — with a live
dashboard for the traffic Studio can no longer see.

Windows, Python 3.10+, no required dependencies.

![The dashboard at /__stats](assets/dashboard.png)

## The problem

Unsloth Studio serves an OpenAI-compatible API on `:8888`. That endpoint is a
Python proxy sitting in front of the real `llama-server` process, and it costs
you on every request. Measured on a dual-GPU Windows box, same model, same
prompt, `llama.cpp` with MTP speculative decoding:

| | via Studio `:8888` | direct to `llama-server` |
|---|---|---|
| short request, median of 10 | 0.302 s | **0.136 s** |
| 400-token generation | 109 tok/s | **127 tok/s** |

Re-measured 2026-09-03 with a 27B reasoning model, 8-token replies, the two
paths interleaved so both saw the same machine state: 531 ms via Studio, 335 ms
direct, median of 10. Different model, same ratio. Streamed generation on that
model was 48.4 tok/s via Studio and 50.1 direct, median of 3, which is within
run-to-run noise. At 50 tok/s the proxy's per-token cost is a small fraction of
each token; the table's 109 vs 127 was measured on a model fast enough for it to
matter. The latency gain does not depend on model speed.

**Why that day's model ran at 50 tok/s, and a warning for anyone with long
agent contexts.** Studio starts `llama-server` with `--kv-unified`, one KV
pool shared by all slots. In that mode a decode pass attends over the whole
occupied pool, not only over its own tokens. Three idle slots were holding
about 100,000 tokens of cached agent context, and a short unrelated request
ran at 64 tok/s, 25 passes per second. With those slots saved and erased, the
same request ran at 131 to 135 tok/s, 52 passes per second. Restoring the
slots brought it back down. `bench/kv_pool.py` reproduces this. The
forwarder's **Largest context** card is the early sign: when it is large and
the server is idle, that context is still in the pool.

**Why the prompt cache misses on a hybrid model, and how to keep it hitting.**
Qwen3.5-family GGUFs (`general.architecture = qwen35`) are hybrids: recurrent
SSM layers with full attention every fourth layer. A recurrent state can be
extended but never rolled back, and Studio starts `llama-server` with
`--ctx-checkpoints 0`, so there is no saved state to roll back to. The cache
is therefore reused only when a new prompt **extends the slot's exact token
sequence**, including the previous reply and its reasoning. Measured on a
15k-token context: an exact extension prefilled 20 tokens in 43 ms; the same
extension with the reasoning dropped, or the same prompt asked twice,
prefilled everything again in about 6,300 ms. Three rules follow for any
client: keep `reasoning_content` in the history you send back, never rewrite
earlier turns, and do not retry a finished turn. The Studio-side fix is to
turn context checkpoints on, at a memory cost per checkpoint per slot.
`bench/agent_loop.py` shows the difference. With the cache hitting, a
20k-token agent turn with a 120-token reply took 1.27 s via Studio and 1.14 s
direct, and the first token arrived after 261 ms via Studio against 107 ms
direct. With the cache missing, both paths took 30 s per turn and the
forwarder could not help, because prefill dominated.

Reproduce it with the scripts in `bench/`. Studio's port needs its API key;
give it as `STUDIO_API_KEY`, or point `STUDIO_API_KEY_FILE` at a file that
holds it. The key is read inside the script and never printed.

```bash
python bench/latency.py       # short request, median round trip
python bench/throughput.py    # streamed 400-token generation, decode tok/s
python bench/kv_pool.py       # decode speed with the KV pool full vs empty
python bench/agent_loop.py    # long cached context + short reply, per turn
```

`kv_pool.py` saves and erases every idle slot, then restores them. Run it only
when nothing else is using the server. It leaves its `bench-kv-pool-slot*.bin`
files in llama-server's `--slot-save-path` directory; delete them afterwards.

An agent loop makes many small calls, so the per-call latency and the
per-token cost both compound.

**So why not just point your client at `llama-server`?** Because Studio starts
it on a *random* free port and re-rolls that port on every model reload. On
the box those numbers came from it was 54966, then 60008, then 55084 — in one
morning. No client can hold a direct endpoint.

## What this does

Listens on one port that never changes, finds Studio's current `llama-server`,
and relays raw TCP.

![Where the forwarder sits](assets/architecture.svg)

- **Raw TCP, not HTTP.** SSE streaming, keep-alive and chunked bodies pass
  through untouched. It reads only each request's first line, to route it.
- **Lazy re-discovery.** When a connect fails it re-scans, so a model reload
  costs one failed connection instead of a config edit.
- **Falls back to Studio.** With no model loaded there is no `llama-server` to
  reach, so it forwards to `:8888` instead. That one request pays the
  overhead, but it triggers Studio's load-on-demand and every later request
  goes direct.
- **Loopback only, with no option to change it.** A forwarder that drops an
  API-key requirement should not be reachable off the machine.

## Install

```bash
pipx install git+https://github.com/kxl3785/omp-forwarder
```

Or from a clone:

```bash
pip install -e .
pip install -e ".[tray]"   # adds pywin32, only needed for --tray
```

Or with no install at all:

```bash
PYTHONPATH=src python -m omp_forwarder
```

The tests need nothing installed and no running `llama-server`:

```bash
python -m unittest
```

## Use

```bash
omp-forwarder                      # listen on 8890
omp-forwarder --tray               # ...with a tray icon
omp-forwarder --port 9000
omp-forwarder --exclude-port 8788  # ignore another llama-server you run
omp-forwarder --upstream-port 55084  # skip discovery, pin the port
```

Then point your client at it. For example, in `~/.omp/agent/models.yml`:

```yaml
providers:
  llama.cpp:
    baseUrl: http://127.0.0.1:8890/v1
```

Any API key your client sends is passed through and ignored — `llama-server`
started by Studio has none.

`run_forwarder.bat` starts it with the tray icon and no console window, and
works straight from a clone with no install — it puts `src\` on `PYTHONPATH`
itself. Make a desktop shortcut to that file and point the shortcut's icon at
`assets/omp-forwarder.ico`.

### Options

| flag | meaning |
|---|---|
| `--port N` | port to listen on (default 8890) |
| `--studio-port N` | Studio's API port, used as the fallback (default 8888) |
| `--upstream-port N` | skip discovery and always use this port |
| `--exclude-port N` | never treat this port as the upstream; repeatable |
| `--tray` | Windows tray icon; needs `pywin32` |

## The dashboard

`http://127.0.0.1:8890/__stats` — and `/__stats.json` for the raw sample.

Once your client bypasses `:8888`, Studio's API panel goes blind to it. This
replaces that panel and adds three numbers it never had:

- **Draft acceptance.** With speculative decoding, this is how you learn that
  speculation quietly fell back to n-gram drafting. Acceptance collapsing
  toward 35% roughly halves your throughput, and nothing else reports it.
- **Prefill rate.** Separates "the model is slow" from "your prompts are long".
- **Prompt-cache hit rate.** A falling hit rate means prompts stopped being
  stable, which costs far more than any decode tuning.

Plus **tokens per pass** (the direct payoff of speculation — 1.00 means it is
buying nothing), in-flight and queued requests, largest context seen, and the
forwarder's own request count, median round trip, and **HTTP 4xx/5xx counts**.
That last one matters: `llama-server`'s `/metrics` has no error counter, so a
500 from a chat template is otherwise completely silent.

**Total tokens** is the forwarder's own tally of prompt and generated tokens.
It exists because every `llama-server` counter restarts at zero when Studio
reloads a model, so a total read straight from `/metrics` only ever covers the
current process. The forwarder outlives reloads: it folds successive
`/metrics` samples into one running total, and a reload loses at most the few
seconds since the last sample. With Session on it counts from when you
clicked, which makes it a per-task or per-day figure for how much work went to
the local model instead of a paid one. It is a count of what `llama-server`
did while the forwarder was running, so requests from Studio's own UI or from
another client land in it too.

Per-day totals survive restarts. They live in `tokens.json` beside the log
(`%LOCALAPPDATA%\omp-forwarder\` with `run_forwarder.bat`; `--tokens-file`
overrides), written within 30 seconds of a change and at exit. The log gets one
line per finished day, at the first request of the next day, plus today's
running figure at start and at exit. The card shows today's total when it
differs from the since-start count, which means the forwarder restarted today.

### Per stream

`/metrics` only aggregates, so with several requests in flight it cannot tell
you which one is slow. The per-stream table reads `/slots` instead and gives
each concurrent request its own row: tok/s, tokens generated, context size,
how much of that context came from cache, and how much of its token budget is
left.

It also separates **prefill** from **decode**, which matters more than it
sounds. A slot showing `decoded = 0` is not idle — it is evaluating its prompt,
and on a long one that is most of the request. The table reports its prefill
rate there rather than a misleading "0 tok/s", and marks slots that are merely
`queued` behind other work.

Measured with two requests running side by side: 45.7 and 51.4 tok/s,
aggregate 97.2. Only decode rates are summed — adding a prefill rate to a
decode rate would produce a number that means nothing.

Every `llama-server` counter is cumulative since the server started, which is
the wrong window when you want to know what just happened. Each section has a
**Session** control that stores a baseline in your browser and subtracts it, so
the figures start from when you clicked. It is a toggle, it is per-browser, and
it changes nothing on the server.

## How discovery works

1. List `llama-server` processes (`tasklist`).
2. Find their listening TCP ports (`netstat -ano`), highest first, skipping
   the forwarder's own port, Studio's port, and anything you excluded.
3. Take the first that answers `GET /health` with 200.
4. Cache it. Re-scan on a failed connect, or from the tray menu.

## Limitations

- **Discovery is Windows-only.** It shells out to `tasklist` and `netstat`.
  The relay itself is portable, so on Linux or macOS pass `--upstream-port`
  and it works fine.
- **The tray needs `pywin32`.** Without it, `--tray` logs a line and runs
  headless.
- **No authentication.** It is loopback-only for that reason. Anything running
  as your user on your machine can reach the model through it — but that was
  already true of `llama-server` itself, which Studio starts with no API key.
- **The 4xx/5xx counter can undercount.** It inspects only reads that begin
  with a status line, rather than tracking HTTP framing, because a relay that
  parsed framing could break streaming. On loopback a status line practically
  never splits across reads. It will not miscount.

## The name

Built for [omp](https://github.com/oh-my-pi/oh-my-pi) (Oh My Pi), which is
where the cost showed up first. Nothing in it is omp-specific — it works for
any OpenAI-compatible client pointed at Unsloth Studio.

## The icon

`assets/omp-forwarder.ico` is committed. Regenerate it with:

```bash
python -m omp_forwarder.make_icon
```

Pure standard library, no Pillow. It prints an ASCII preview so you can check
legibility at 16px without opening the file. Two notes if you adapt it: the
frames are uncompressed BMP rather than PNG-in-ICO, because GDI+ renders PNG
frames as per-pixel-alpha noise when the file is loaded in-process; and there
is one drawing at every size, so a 16px tray icon is the same logo as the
256px one.

## License

MIT
