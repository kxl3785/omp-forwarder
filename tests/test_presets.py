"""Model presets: assign a measured launch recipe to this lane's GPU.

A preset never picks a card; the lane's --gpu does. Assigning unloads what
the lane fronts, launches the recipe with {gpu}/{port}/{name} filled, and
points the lane's upstream at the new port. Nothing here starts a process:
_spawn_host, _run_host, _run_wsl and _spawn_wsl are all replaced."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from unittest import mock

from omp_forwarder import forwarder as fwd
from omp_forwarder import stats

from .helpers import ForwarderCase, RelayCase, raw_request


def _completed(stdout: str = "", rc: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=rc,
                                       stdout=stdout, stderr="")


def _body(out: bytes) -> dict:
    return json.loads(out.split(b"\r\n\r\n", 1)[1])


PRESETS = {
    "llama-tune": {"kind": "process", "port": "4950{gpu}",
                   "cmd": "bash launch.sh {gpu} {port}"},
    "sglang-nothink": {"kind": "container", "port": "3000{gpu}",
                       "distro": "Ubuntu-24.04", "container": "sgl{gpu}",
                       "run": "docker run -d --name {name} -p {port}:30000 "
                              "-e CUDA_VISIBLE_DEVICES={gpu} img serve "
                              "--kw '{\"enable_thinking\": false}'"},
    "broken": {"kind": "spaceship"},
}


class PresetFileTests(ForwarderCase):

    def setUp(self):
        super().setUp()
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(self.tmp, ignore_errors=True))
        fwd.TOKENS_FILE = os.path.join(self.tmp, "tokens.json")

    def test_loads_only_valid_kinds(self):
        with open(os.path.join(self.tmp, "presets.json"), "w", encoding="utf-8") as fh:
            json.dump(PRESETS, fh)
        fwd._load_presets()
        self.assertEqual(sorted(fwd._presets), ["llama-tune", "sglang-nothink"])

    def test_missing_file_means_no_presets(self):
        fwd._load_presets()
        self.assertEqual(fwd._presets, {})

    def test_explicit_path_wins(self):
        p = os.path.join(self.tmp, "elsewhere.json")
        with open(p, "w", encoding="utf-8") as fh:
            json.dump({"only": PRESETS["llama-tune"]}, fh)
        fwd.PRESETS_FILE = p
        fwd._load_presets()
        self.assertEqual(list(fwd._presets), ["only"])

    def test_render_keeps_json_braces(self):
        out = fwd._render(PRESETS["sglang-nothink"]["run"], 1, 30001, "sgl1")
        self.assertIn("--name sgl1", out)
        self.assertIn("-p 30001:30000", out)
        self.assertIn("CUDA_VISIBLE_DEVICES=1", out)
        self.assertIn('{"enable_thinking": false}', out)

    def test_port_template(self):
        self.assertEqual(fwd._preset_port(PRESETS["llama-tune"], 0), 49500)
        self.assertEqual(fwd._preset_port(PRESETS["sglang-nothink"], 1), 30001)
        self.assertEqual(fwd._preset_port({"port": 8080}, 1), 8080)


class AssignTests(ForwarderCase):

    def setUp(self):
        super().setUp()
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(self.tmp, ignore_errors=True))
        fwd.TOKENS_FILE = os.path.join(self.tmp, "tokens.json")
        fwd.LISTEN_PORT = 8899
        fwd._presets = {k: v for k, v in PRESETS.items() if k != "broken"}
        fwd.FWD_GPU = 1

    def test_no_gpu_refuses(self):
        fwd.FWD_GPU = None
        self.assertEqual(fwd._assign_preset("llama-tune"), ("no-gpu", None))

    def test_unknown_preset(self):
        self.assertEqual(fwd._assign_preset("nope"), ("unknown-preset", None))

    def test_process_preset_unloads_then_spawns_on_the_lane_gpu(self):
        # Currently fronting a process on 41999 owned by pid 4242.
        fwd._upstream = 41999
        fwd._port_owner[41999] = "4242"
        killed, spawned = [], []
        with mock.patch.object(fwd, "_run_host",
                               side_effect=lambda a, timeout=10.0: killed.append(a) or _completed()), \
                mock.patch.object(fwd, "_spawn_host",
                                  side_effect=lambda a: spawned.append(a) or mock.Mock()), \
                mock.patch.object(fwd, "_git_bash", return_value=None):
            status, port = fwd._assign_preset("llama-tune")
        self.assertEqual((status, port), ("loading", 49501))
        self.assertEqual(len(killed), 1)
        self.assertIn("4242", killed[0])
        self.assertEqual(spawned[0][:2], ["bash", "launch.sh"])
        self.assertEqual(spawned[0][2:], ["1", "49501"])
        self.assertEqual(fwd.FORCED_UPSTREAM, 49501)
        # The lane points at the new port at once; nothing waits for a request.
        self.assertEqual((fwd._upstream, fwd._upstream_kind), (49501, "explicit"))
        self.assertFalse(fwd._upstream_healthy)
        self.assertEqual(fwd._preset, "llama-tune")
        self.assertFalse(fwd._operator_stopped)

    def test_stop_works_right_after_assign(self):
        # Live run 2026-09-05: assign cleared _upstream, so stop had no port
        # to key the PID lookup on and answered "no-process" while a 27B sat
        # on the card. The netstat pass must find the new port's owner.
        killed = []
        with mock.patch.object(fwd, "_spawn_host", return_value=mock.Mock()),                 mock.patch.object(fwd, "_run_host",
                                  side_effect=lambda a, timeout=10.0: killed.append(a) or _completed()),                 mock.patch.object(fwd, "_port_pid", side_effect=lambda port: "777" if port == 49501 else None):
            fwd._assign_preset("llama-tune")
            status = fwd._control_action("stop")
        self.assertEqual(status, "stopped")
        self.assertTrue(any("777" in a for a in killed), killed)
        self.assertTrue(fwd._operator_stopped)

    def test_container_preset_stops_old_container_and_runs_new(self):
        fwd.WSL_DISTRO, fwd.CONTAINER_NAME = "Ubuntu-24.04", "sglold"
        seen = []
        with mock.patch.object(fwd, "_run_wsl",
                               side_effect=lambda a, timeout=10.0: seen.append(a) or _completed()), \
                mock.patch.object(fwd, "_spawn_wsl", return_value=mock.Mock()):
            status, port = fwd._assign_preset("sglang-nothink")
        self.assertEqual((status, port), ("loading", 30001))
        cmds = [" ".join(a) for a in seen]
        self.assertTrue(any("docker stop sglold" in c for c in cmds), cmds)
        self.assertTrue(any("docker rm -f sgl1" in c for c in cmds), cmds)
        run = [a for a in seen if "bash" in a and "-c" in a][0]
        self.assertIn("--name sgl1", run[-1])
        self.assertIn("-p 30001:30000", run[-1])
        self.assertEqual((fwd.WSL_DISTRO, fwd.CONTAINER_NAME), ("Ubuntu-24.04", "sgl1"))
        self.assertEqual(fwd.FORCED_UPSTREAM, 30001)

    def test_assignment_persists_and_is_adopted(self):
        with mock.patch.object(fwd, "_spawn_host", return_value=mock.Mock()), \
                mock.patch.object(fwd, "_run_host", return_value=_completed()):
            fwd._assign_preset("llama-tune")
        fwd._preset, fwd._saved_state = None, {}
        fwd._load_latch()
        self.assertEqual(fwd._preset, "llama-tune")
        self.assertEqual(fwd._saved_state.get("upstream_port"), 49501)
        self.assertIsNone(fwd._saved_state.get("container"))


class AssignEndpointTests(RelayCase):

    def setUp(self):
        super().setUp()
        fwd._control_token = "tok"
        fwd._presets = {k: v for k, v in PRESETS.items() if k != "broken"}
        fwd.FWD_GPU = 0

    def _post(self, q: str) -> bytes:
        return raw_request(self.port, [f"POST /__control?token=tok&{q} HTTP/1.1\r\n\r\n".encode()])

    def test_assign_process_preset(self):
        with mock.patch.object(fwd, "_spawn_host", return_value=mock.Mock()), \
                mock.patch.object(fwd, "_run_host", return_value=_completed()):
            out = self._post("action=assign&preset=llama-tune")
        self.assertTrue(out.startswith(b"HTTP/1.1 200"), out[:80])
        b = _body(out)
        self.assertEqual((b["status"], b["preset"], b["port"]), ("loading", "llama-tune", 49500))
        self.assertIn(b"Connection: close", out)

    def test_assign_without_gpu_is_409(self):
        fwd.FWD_GPU = None
        out = self._post("action=assign&preset=llama-tune")
        self.assertTrue(out.startswith(b"HTTP/1.1 409"), out[:80])

    def test_assign_unknown_is_400_and_lists_presets(self):
        out = self._post("action=assign&preset=nope")
        self.assertTrue(out.startswith(b"HTTP/1.1 400"), out[:80])
        self.assertEqual(_body(out)["presets"], ["llama-tune", "sglang-nothink"])

    def test_presets_alone_make_the_lane_controllable(self):
        out = raw_request(self.port, [b"GET /__control?token=tok&action=assign HTTP/1.1\r\n\r\n"])
        self.assertTrue(out.startswith(b"HTTP/1.1 405"), out[:80])


class SnapshotPresetTests(ForwarderCase):

    def test_fields(self):
        fwd._presets = {"b": {"kind": "process"}, "a": {"kind": "process"}}
        fwd._preset = "a"
        fwd._upstream_healthy = False
        fwd._operator_stopped = False
        s = stats.snapshot(fwd, dict(fwd._stats))
        self.assertEqual(s["presets"], ["a", "b"])
        self.assertEqual(s["preset"], "a")
        self.assertTrue(s["loading"])
        fwd._upstream_healthy = True
        self.assertFalse(stats.snapshot(fwd, dict(fwd._stats))["loading"])


class HostArgvTests(ForwarderCase):
    """The launch command becomes argv the way a shell would read it, and a
    bare `bash` means Git's bash. Live 2026-09-05: `bash launch.sh` under a
    plain Windows PATH ran WSL's System32 bash, which cannot open a Windows
    path, so assign answered "loading" twice and no server ever started;
    the quoted Git path then kept its quotes through shlex(posix=False) and
    CreateProcess could not find it either."""

    def test_quoted_path_with_spaces_is_one_token(self):
        with mock.patch.object(fwd, "_git_bash", return_value=None):
            argv = fwd._host_argv('"C:/Program Files/Git/usr/bin/bash.exe" C:/x/launch.sh 0 49500')
        self.assertEqual(argv, ["C:/Program Files/Git/usr/bin/bash.exe",
                                "C:/x/launch.sh", "0", "49500"])

    def test_backslash_paths_survive(self):
        with mock.patch.object(fwd, "_git_bash", return_value=None), \
                mock.patch.object(fwd.os, "name", "nt"):
            argv = fwd._host_argv(r'"C:\Program Files\Git\usr\bin\bash.exe" C:\x\launch.sh 0 49500')
        self.assertEqual(argv[0], "C:/Program Files/Git/usr/bin/bash.exe")
        self.assertEqual(argv[1:], ["C:/x/launch.sh", "0", "49500"])

    def test_bare_bash_means_git_bash_when_present(self):
        with mock.patch.object(fwd, "_git_bash", return_value="C:/git/bash.exe"):
            self.assertEqual(fwd._host_argv("bash launch.sh 1 49501")[0], "C:/git/bash.exe")
        with mock.patch.object(fwd, "_git_bash", return_value=None):
            self.assertEqual(fwd._host_argv("bash launch.sh 1 49501")[0], "bash")

    def test_failed_spawn_is_reported_not_raised(self):
        with mock.patch.object(fwd.subprocess, "Popen", side_effect=OSError("nope")):
            self.assertIsNone(fwd._spawn_host(["no-such-exe"]))


class SpawnFailedAssignTests(ForwarderCase):

    def setUp(self):
        super().setUp()
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(self.tmp, ignore_errors=True))
        fwd.TOKENS_FILE = os.path.join(self.tmp, "tokens.json")
        with open(os.path.join(self.tmp, "presets.json"), "w", encoding="utf-8") as fh:
            json.dump(PRESETS, fh)
        fwd._load_presets()
        fwd.FWD_GPU = 1

    def test_assign_answers_spawn_failed(self):
        with mock.patch.object(fwd, "_run_host", return_value=_completed()), \
                mock.patch.object(fwd, "_spawn_host", return_value=None), \
                mock.patch.object(fwd, "_git_bash", return_value=None):
            self.assertEqual(fwd._assign_preset("llama-tune"), ("spawn-failed", 49501))

    def test_assign_rereads_the_presets_file(self):
        # a recipe added after startup is usable without a restart
        extra = dict(PRESETS)
        extra["late"] = {"kind": "process", "port": "4960{gpu}", "cmd": "bash late.sh {gpu} {port}"}
        with open(os.path.join(self.tmp, "presets.json"), "w", encoding="utf-8") as fh:
            json.dump(extra, fh)
        with mock.patch.object(fwd, "_run_host", return_value=_completed()), \
                mock.patch.object(fwd, "_spawn_host", return_value=mock.Mock()), \
                mock.patch.object(fwd, "_git_bash", return_value=None):
            self.assertEqual(fwd._assign_preset("late"), ("loading", 49601))


#: A container preset that carries measured costs, so the lane can size its
#: KV window to the card instead of to a number written once. The figures are
#: round for arithmetic, not real: the live ones live in presets.json.
SIZED = {
    "kind": "container", "port": "4960{gpu}", "distro": "Ubuntu-24.04",
    "container": "eng{gpu}",
    "run": "docker run -d --name {name} -p {port}:8080 img serve "
           "--device {gpu} --max-context {kv} --kv-capacity {kv}",
    "sizing": {"ladder": [262144, 196608, 131072], "weights_mib": 20000,
               "workspace_mib": 500, "overhead_mib": 1000,
               "kib_per_token": 35, "tenant_floor_mib": 0},
}


def _sized(**over) -> dict:
    p = json.loads(json.dumps(SIZED))
    p["sizing"].update(over)
    return p


class KvSizingTests(ForwarderCase):
    """The KV window is chosen at launch, from what the card has free.

    An engine that sizes its cache at load and never resizes has to be told
    the right number on its command line, and the right number changes with
    whatever else is resident. These tests pin the arithmetic and the ladder;
    the engine's own refusal is covered by the step-down test."""

    def setUp(self):
        super().setUp()
        fwd.FWD_GPU = 1
        self.ran: list[list[str]] = []

    def _launch(self, preset: dict, free: int, total: int = 32768,
                settle="running"):
        fwd._presets = {"eng": preset}
        smi = _completed(f"{free}, {total}\n")
        with mock.patch.object(fwd.time, "sleep"), \
                mock.patch.object(fwd, "_run_host", return_value=smi), \
                mock.patch.object(fwd, "_run_wsl",
                                  side_effect=lambda a, timeout=10.0:
                                  self.ran.append(a) or _completed()), \
                mock.patch.object(fwd, "_spawn_wsl", return_value=mock.Mock()), \
                mock.patch.object(fwd, "_await_container",
                                  side_effect=(settle if isinstance(settle, list)
                                               else None),
                                  return_value=(None if isinstance(settle, list)
                                                else settle)):
            return fwd._assign_preset("eng")

    def _run_line(self) -> str:
        return [a for a in self.ran if "bash" in a and "-c" in a][-1][-1]

    def test_an_empty_card_takes_the_top_rung(self):
        # 32,000 free: 768 resident, no tenant floor, so the budget is
        # 32000-20000-500-1000 = 10,500 MiB, or about 307k tokens at 35 KiB.
        self._launch(_sized(), free=32000)
        self.assertEqual(fwd._kv_plan["tokens"], 262144)
        self.assertIn("--max-context 262144", self._run_line())
        self.assertIn("--kv-capacity 262144", self._run_line())

    def test_a_busy_card_takes_a_smaller_rung(self):
        # 26,000 free leaves 4,500 MiB for the cache: 131k, not 262k.
        self._launch(_sized(), free=26000)
        self.assertEqual(fwd._kv_plan["tokens"], 131072)
        self.assertIn("--max-context 131072", self._run_line())

    def test_the_absent_tenant_keeps_its_floor(self):
        # The same empty card, but another program needs 5 GiB to come back.
        # Only the shortfall is held: 5000 - 768 resident = 4,232 MiB.
        self._launch(_sized(tenant_floor_mib=5000), free=32000)
        self.assertEqual(fwd._kv_plan["tokens"], 131072)
        self.assertEqual(fwd._kv_plan["reserve_mib"], 4232)

    def test_a_resident_tenant_is_not_reserved_twice(self):
        # The tenant is already loaded, so its 6,768 MiB is missing from
        # free. Holding its floor again would cost a rung for nothing.
        self._launch(_sized(tenant_floor_mib=5000), free=26000)
        self.assertEqual(fwd._kv_plan["reserve_mib"], 0)
        self.assertEqual(fwd._kv_plan["others_mib"], 6768)

    def test_no_rung_fits_so_the_smallest_is_attempted(self):
        # 3,500 MiB of budget buys 102k tokens and the ladder stops at 131k.
        # Launching anyway puts the engine's own refusal in its log, which is
        # worth more than a lane that was never started.
        self._launch(_sized(), free=25000)
        self.assertEqual(fwd._kv_plan["tokens"], 131072)
        self.assertIn("affords no rung", fwd._kv_plan["why"])

    def test_an_unreadable_card_takes_the_smallest_window(self):
        fwd._presets = {"eng": _sized()}
        with mock.patch.object(fwd, "_run_host", return_value=None), \
                mock.patch.object(fwd, "_run_wsl",
                                  side_effect=lambda a, timeout=10.0:
                                  self.ran.append(a) or _completed()), \
                mock.patch.object(fwd, "_spawn_wsl", return_value=mock.Mock()), \
                mock.patch.object(fwd, "_await_container", return_value="running"):
            fwd._assign_preset("eng")
        self.assertEqual(fwd._kv_plan["tokens"], 131072)
        self.assertIsNone(fwd._kv_plan["free_mib"])

    def test_a_refused_window_steps_down_one_rung(self):
        # The estimate can overshoot: the engine knows its own reservation
        # and this does not. A refusal must cost a rung, not the lane.
        self._launch(_sized(), free=32000, settle=["exited", "running"])
        self.assertEqual(fwd._kv_plan["tokens"], 196608)
        self.assertIn("stepped down", fwd._kv_plan["why"])
        self.assertIn("--max-context 196608", self._run_line())

    def test_it_stops_after_two_step_downs(self):
        self._launch(_sized(), free=32000,
                     settle=["exited", "exited", "exited"])
        self.assertEqual(fwd._kv_plan["tokens"], 131072)
        runs = [a for a in self.ran if "bash" in a and "-c" in a]
        self.assertEqual(len(runs), 3)

    def test_a_preset_without_sizing_is_launched_unchanged(self):
        fwd._presets = {"eng": {k: v for k, v in SIZED.items() if k != "sizing"}}
        with mock.patch.object(fwd, "_run_host", return_value=_completed()), \
                mock.patch.object(fwd, "_spawn_wsl", return_value=mock.Mock()), \
                mock.patch.object(fwd, "_run_wsl",
                                  side_effect=lambda a, timeout=10.0:
                                  self.ran.append(a) or _completed()):
            fwd._assign_preset("eng")
        self.assertEqual(fwd._kv_plan, {})
        self.assertIn("{kv}", self._run_line())

    def test_a_process_preset_clears_a_stale_window(self):
        fwd._kv_plan = {"tokens": 262144}
        fwd._presets = {"proc": {"kind": "process", "port": "4950{gpu}",
                                 "cmd": "bash launch.sh {gpu} {port}"}}
        with mock.patch.object(fwd, "_spawn_host", return_value=mock.Mock()), \
                mock.patch.object(fwd, "_run_host", return_value=_completed()):
            fwd._assign_preset("proc")
        self.assertEqual(fwd._kv_plan, {})

    def test_await_container_returns_as_soon_as_it_exits(self):
        # The refusal lands about six seconds in. Waiting out the grace
        # would add six seconds to every failed button press.
        with mock.patch.object(fwd.time, "sleep"), \
                mock.patch.object(fwd, "_run_wsl",
                                  return_value=_completed("false\n")):
            self.assertEqual(fwd._await_container("D", "c"), "exited")


class GpuMemSettleTests(ForwarderCase):
    """A card does not give its memory back the instant a container dies."""

    def test_it_waits_for_two_readings_that_agree(self):
        # docker rm -f returns long before the driver releases 20 GiB.
        # Measured 2026-09-13: a lane re-assigned on GPU 0 read 4,595 MiB
        # free with its own outgoing container still resident, computed a
        # negative budget, and fell to the smallest rung on a card that was
        # about to be nearly empty.
        reads = ["4595, 32768\n", "18000, 32768\n",
                 "31900, 32768\n", "32000, 32768\n"]
        with mock.patch.object(fwd.time, "sleep"), \
                mock.patch.object(fwd, "_run_host",
                                  side_effect=[_completed(r) for r in reads]):
            self.assertEqual(fwd._gpu_mem_settled(1), (32000, 32768))

    def test_a_steady_card_is_read_twice_and_no_more(self):
        calls = []
        with mock.patch.object(fwd.time, "sleep"), \
                mock.patch.object(fwd, "_run_host",
                                  side_effect=lambda a, timeout=10.0:
                                  calls.append(a) or _completed("30000, 32768\n")):
            self.assertEqual(fwd._gpu_mem_settled(0), (30000, 32768))
        self.assertEqual(len(calls), 2)

    def test_an_unreadable_card_gives_up_at_once(self):
        with mock.patch.object(fwd.time, "sleep"), \
                mock.patch.object(fwd, "_run_host", return_value=None):
            self.assertIsNone(fwd._gpu_mem_settled(0))
