# OMP Forwarder

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

An agent loop makes many small calls, so the per-call latency and the
per-token cost both compound.

**So why not just point your client at `llama-server`?** Because Studio starts
it on a *random* free port and re-rolls that port on every model reload. On
the box those numbers came from it was 54966, then 60008, then 55084 — in one
morning. No client can hold a direct endpoint.

## What this does

Listens on one port that never changes, finds Studio's current `llama-server`,
and relays raw TCP.

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
