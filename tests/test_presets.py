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
                                  side_effect=lambda a: spawned.append(a) or mock.Mock()):
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
