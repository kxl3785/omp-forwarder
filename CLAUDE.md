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

**Never relay to Studio for a client that lacks Studio's key.** Studio's API
requires one; a client pointed here sends whatever key it likes, because
`llama-server` ignores keys. So the old "fall back to :8888" path handed omp a
**401**, which a client reads as "your config is wrong, stop trying" when the
truth is "the model is loading". It cost a real outage on 2026-09-03. The
forwarder now waits `--wait-for-model` seconds for a server to appear, then
answers **503 with Retry-After**. `--studio-fallback` restores the old
behaviour for a client that does hold the key.

**Container upstreams (SGLang in WSL), added 2026-09-05.** WSL2 shuts the
distro down when its last `wsl.exe` client exits, and Docker inside it then
SIGTERMs every container — so `--container` mode holds a `sleep infinity`
child for the forwarder's lifetime and stops it in a `finally`. SGLang has no
`/metrics` and no `/slots`: the dashboard light now means `/health`, and
`metrics_available` says whether the llama-server-only cards have data;
SGLang's facts come from `/get_server_info`, llama-server's from `/props`,
never on the request path. `/__control` is POST plus a launch-time token and
there are no CORS headers anywhere — a foreign page cannot read
`/__stats.json`, so it cannot learn the token, and GET never mutates. Routing
reads only the first line and query string, so the relay still parses no
bodies. `--name`/`--peer` exist so two lane dashboards tell each other apart.
**The dashboard's stop latches.** Auto-start restarted an exited container
within a minute of the operator stopping it, which turned "unload the GPU"
into "reload the GPU". `_operator_stopped` is set by stop and cleared by
start/restart, and `_poll_container_status` honours it. On a process lane
stop terminates the PID from the netstat scan; start needs `--upstream-cmd`.

**Discovery sees two kinds of upstream, and re-evaluates.** Executable
matching cannot see a container, so candidates also come from
`--candidate-port` and from `docker port <container>` in container mode; every
candidate gets the same `/health` probe, and `--prefer` (default
`llama-server`) decides when both kinds are healthy. The first version
evaluated that preference once, at startup: in the live test a llama-server
that appeared beside a healthy container was never noticed. The monitor thread
now recomputes the choice every 10 s and switches *new* connections when a
preferred healthy upstream exists; existing connections are never touched.
`--upstream-port` still overrides all of it, and nothing falls back to :8888.
For the dashboard's model cards on an SGLang lane, launch SGLang with
`--enable-metrics`; without it `/metrics` is 404 and the cards say so.

**Discovery cannot tell two `llama-server` processes apart by port.** It takes
the highest healthy one. RadHelper runs its own 4B model on a llama-server,
and excluding Studio's port made discovery silently select that one — a 4B
radiology model answering coding requests, with nothing in the reply to say
so. Its port moves (8788, then 8799), so a hard-coded `--exclude-port` goes
stale. `--upstream-exe .unsloth` is the fix: it matches on the executable
path, which is the only stable discriminator, and `_exe_path` reads it with
`ctypes` in 0.03 ms rather than spawning PowerShell (252 ms). `wmic` is gone
from Windows 11; do not reach for it.

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

**Presets are the only launch knowledge the forwarder holds, and they live outside the repo.** `presets.json` beside tokens.json carries this machine's paths and CUDA quirks; the code only fills `{gpu}`/`{port}`/`{name}` with `str.replace`, never `str.format`, because a docker run line has JSON braces of its own. Assign switches the lane by setting `FORCED_UPSTREAM` **and `_upstream`** to the new port at once, as a static `--upstream-port` lane is. The first version cleared `_upstream` for discover() to re-adopt, and discover() runs only on a request: the health sampler probed nothing, the page read `loading` through a served completion, and stop found no PID because the lookup keys on `_upstream`. A container preset on a lane that started as a process lane also has to start the container monitor itself (`_ensure_container_monitor`), or the dashboard reads `unknown` for a running container. The assignment and the latch share `control-<port>.json` and are adopted at startup unless the command line says otherwise.

**The lane relay is the only cross-lane write, and the peer token never reaches a page.** `_sample_peers` puts a peer's `control_token` in `_peer_tokens`, not in `_peer_state`, because the snapshot copies `_peer_state` verbatim to the browser. `lane=<port>` on `/__control` is gated by the local token, limited to `PEERS`, and calls `_peer_control` (the one seam) with the peer's token. Keep it that way: a page that could read a peer's token would make the no-CORS rule pointless.

**A connect to a closed loopback port TIMES OUT on this Windows box; it is not refused.** Measured 2026-09-05: 2.0 s to fail against a dead 49500. Every reader that touches the upstream on the request path pays it, so an unloaded lane's `/__stats.json` took 4 s (metrics + slots), the page crawled, and a peer's read of it never succeeded. `stats.snapshot` therefore skips `upstream_metrics`/`upstream_slots` once `_health_sampled` is set and `_upstream_healthy` is False. Never add a request-path read of the upstream without that gate.

**The header's model name is the sampler's job, and a dash is not a name.** `upstream_model` answers `-` when the server is not ready; the old request-path fill stored that and never asked again, so a page opened during a load showed no model for the life of the process. `_sample_model` retries every tick while the upstream is healthy and the name is `""` or `-`. That name, beside a green light and a `serving` row, is the operator's proof that the right model loaded; keep it live.

**A lane owns one GPU, and discovery respects that.** With two lanes and `--upstream-exe` matching both cards' llama-servers, the GPU 0 lane with nothing assigned discovered the server the GPU 1 lane had just loaded: highest healthy port wins. Its traffic would have crossed cards and its header `stop` would have killed the other card's model. `_taken_ports` now removes from discovery every port a peer fronts (from the peer snapshot's `upstream`) and every preset port that maps to another GPU. Studio's own server on 49500 therefore belongs to the GPU 0 lane by the `4950{gpu}` convention.

**A container preset must publish on 0.0.0.0, never on WSL's loopback.** The
forwarder runs on Windows and the container runs in the WSL distro, and WSL2
forwards a published port to Windows localhost only when it is bound to
`0.0.0.0`. Measured 2026-09-12: `-p 127.0.0.1:28080:8080` answered `/health`
from inside WSL and was refused from Windows, which looks exactly like a dead
upstream. Write `-p {port}:8080`, as the SGLang preset does.

**A container also dies when the last `wsl.exe` client exits** — the same
WSL2 shutdown that `--container` mode holds off with its `sleep infinity`
child. Anything that starts a preset container outside the forwarder needs
its own keepalive, or the model unloads a minute later with exit code 0 and
no error anywhere.

**Preset notes carry the measurement, not just the command.** The 2026-09-12
campaign added `ninfer-nvfp4` and `ninfer-coldfusion` (NInfer, one engine per
card, port `4960{gpu}` because `llama-tune` owns `4950{gpu}`). Their notes
record what was measured: the NVFP4 lane's KV pool holds 232,768 tokens, so
two 80k conversations fit and a third evicts both; the Cold-Fusion lane is for
thinking-on work, where it finished the suite in 194 s against 287 s for the
official weights. Tensor parallelism across the two cards is NOT a preset and
must not become one: these GeForce cards grant no peer access, the collectives
stage through host memory at 29.5-51.6 us per exchange with 128 exchanges per
token, and `--tp 2` measured about half the speed of one card while losing
prefix reuse entirely.

**The default `/__stats.json` is the fleet; `?self=1` is the lane.** `merge_snapshots` sums cumulative counters BEFORE the page differences them, which is what makes the page's rate logic produce fleet rates unchanged. Per-stream ids become `<port>:<slot>` because the page keys its rate history by id. Peers are read on the page's poll, but only the ones the sampler last saw reachable (a dead port costs 2 s here), and always with `?self=1`, or two lanes fold each other in without end. The peer's `control_token` is dropped before the merge.

**`merge_snapshots` must answer `days` newest first, because `recent_days` does.** The usage page reads `days[0]` as today and `days[0:7]` as the last week. Sorted ascending, the fleet page labelled today with the oldest day on record, wrote the live figure into that row, and summed the seven OLDEST days as "last 7 days". One `reverse=True`.

**Nothing on either page links to another lane.** Both pages read the fleet snapshot, so every lane is already on the page you have open. The peer pill in the header, the lane row's port link and the GPU panel's "this lane" marker all invited the operator to go looking for the other card's copy of the same page, which is how two pages got made out of one again. Every lane row now reads the same and names itself; `tests/test_dashboard.py` asserts no `http://127.0.0.1:` appears in either page.

**A session baseline outlives the counters it was taken from.** It lives in `localStorage`, so it survives a reload, a forwarder restart and a model reload -- and every one of those puts the counters back at zero. The subtraction then clamps at zero and the whole section reads "0" through real traffic: measured 2026-09-12, requests 0 and connections 0 beside a 21.9 s median round trip. `dropStaleBaselines` drops a baseline any counter has fallen below and the section returns to lifetime.

**NInfer publishes no token counters, so the tally reads its log.** `_tally_tokens` never fires on such a lane and the Tokens card read zero through a day of real traffic. `_tally_ninfer_tokens` folds each `request_done` in once, keyed by request id AND timestamp because a restarted engine numbers its requests from one again. `prompt_tokens` there INCLUDES the cached prefix and llama-server reports the two disjoint, so the cached part is subtracted out rather than counted twice. The first read of a log is a baseline: that tail was not our traffic.

**A lane sizes its KV window at launch, because the card's other tenants move.** NInfer sizes its cache once, at load, and never resizes, so the number has to be right on the command line -- and the right number changes. Measured 2026-09-13: both lanes refused to start, each about a gigabyte short, because a radiology tool had loaded twenty seconds earlier. `_plan_kv` reads `nvidia-smi` fresh (`_gpu_state` is 10 s old, and the lane was unloaded in between), subtracts the preset's measured costs, and picks the largest rung of a LADDER the card affords. A ladder, not an exact fit: sizing to the free byte would advertise a different context every restart, and an agent that planned around 262k would silently get 131k. `tenant_floor_mib` holds room for a tenant that is DOWN at launch, and only the shortfall -- whatever the tenant already holds is missing from `free` anyway, and reserving it twice costs a rung for nothing. The estimate can still overshoot, so a refusal steps down one rung and retries, twice at most; `_await_container` watches for the exit rather than the listen, because the refusal lands at six seconds and the listen at twelve. The floor is a knob, not a policy: it is 0 on this box from 2026-09-13, because the window is meant to follow the VRAM that is actually allocatable at load. A non-zero floor made the budget constant instead -- both cards afforded exactly 161,089 tokens whether the tenant ran or not -- and the operator wants the opposite. At 0 the two cards diverge as they should: 262,144 on the empty card and 196,608 on the one that also drives the displays. Give the ladder low rungs as well as high ones. With no floor a card can be genuinely short, and a ladder that stops at its nominal window has nothing to step down to: on 2026-09-13 both lanes refused outright, one over a CUDA-graph allowance and one for lack of room, and neither had a smaller rung to try. With 98,304 and 65,536 added, the graph failure stepped down and served. The other card did not: a lane needs about 3 GiB of runtime at the smallest rung, and 692 MiB was all that was left. No rung saves a card that another program has filled. The chosen window and its arithmetic ride the snapshot as `kv_plan` and land on the Lanes panel, since no engine here reports its window over HTTP.

**A container publishes a port that has never been published before.** WSL2 forwards a published container port to Windows localhost, and on 2026-09-13 it kept the forward twice after the container was gone. The next launch republished the same `4960{gpu}`, netstat showed TWO listeners for it, and every connect went to the dead one: the engine answered `/health` with 200 inside the distro while the lane read `loading` for as long as anyone watched. Restarting the container did not clear it. Only `wsl --shutdown` did, and that unloads the OTHER card's model as well, which is why a WSL restart must never be part of a model switch -- two lanes exist so that one can change while the other keeps serving. A preset's `"port": "auto"` now means the forwarder picks a port with no listener (`_free_container_port`, from one netstat pass) and fills `{port}` with it. A port that has never been published has no forward to go stale. `_preset_port` stays pure and answers None for `auto`, so a caller that only wants to know which card a recipe belongs to cannot burn a port as a side effect; `_launch_port` is the one that allocates.


**Two engines share one WSL virtual machine, so a model load starves the other lane.** One VM, one Docker daemon, one disk. Measured 2026-09-13 across four switches of the neighbouring card: the serving lane never went unhealthy and the forwarder never refused a request (`unavailable` stayed 0), but its median round trip went from 0.2 s at rest to 7.2 s, and two requests sat in the engine's admission queue past 30 s and came back 503 -- `inference request expired while waiting for admission`. The preset now passes `--pending-timeout-ms 120000` rather than the engine's 30,000 default, for the same reason the relay answers 503-with-Retry-After instead of Studio's 401: a client that waits gets its answer, and a client that gets an error stops trying. The cost is that a genuinely stuck lane takes two minutes to say so. Do NOT reach for a WSL restart to fix a lane: it unloads the OTHER card's model as well.

**A third rate bug, same shape as the first two: NInfer's live decode rate counts prefill against it.** The engine's periodic `throughput` record averages over the whole interval, so a window that straddles a prefill is mostly prefill -- true of the window, badly wrong about the stream. Measured 2026-09-13 over an hour of real traffic: on the GPU 0 lane, windows carrying a prefill read 44.0 tok/s while the requests finishing inside them ran at 184.9, and 570 of 588 windows carried one. The card was almost always showing the diluted figure, which is why it never agreed with the Recent list beside it. `ninfer_log_stats` now takes the live decode rate from decode TIME -- generated tokens over decode seconds, across requests finished in the last `NINFER_LIVE_WINDOW_S` -- and falls back to the engine's record only when that window carried no prefill. Prefill needs no such care: it is measured over prefill work. Before you trust any engine's own rate, check what its denominator includes.
