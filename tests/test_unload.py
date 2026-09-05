"""Unload control: stop means the same on every lane, and it sticks.

A container lane stops with docker; a process lane stops by terminating the
PID the netstat scan attributed to the upstream port. Either way the
operator's stop latches, so the container monitor cannot auto-start what
was just unloaded. start/restart on a process lane exist only when
--upstream-cmd says how. Nothing here starts a real process: every
subprocess seam is replaced."""
from __future__ import annotations

import json
import subprocess
from unittest import mock

from omp_forwarder import forwarder as fwd
from omp_forwarder import stats

from .helpers import ForwarderCase, RelayCase, raw_request


def _completed(stdout: str = "", rc: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=rc,
                                       stdout=stdout, stderr="")


def _body(out: bytes) -> dict:
    return json.loads(out.split(b"\r\n\r\n", 1)[1])


class OperatorLatchTests(ForwarderCase):
    """The stop latch is what makes an unload stay unloaded."""

    def setUp(self):
        super().setUp()
        fwd.WSL_DISTRO = "Ubuntu-24.04"
        fwd.CONTAINER_NAME = "sgl"
        fwd._upstream = None
        fwd._container_last_start = 0.0

    def _calls(self, latched: bool) -> list:
        fwd._operator_stopped = latched
        seen = []

        def fake(args, timeout=10.0):
            seen.append(args)
            return _completed("exited")

        with mock.patch.object(fwd, "_run_wsl", side_effect=fake):
            fwd._poll_container_status()
        return [a for a in seen if "start" in a]

    def test_auto_start_fires_when_not_latched(self):
        self.assertEqual(len(self._calls(latched=False)), 1)

    def test_auto_start_suppressed_while_latched(self):
        self.assertEqual(self._calls(latched=True), [])

    def test_container_stop_sets_latch_and_start_clears_it(self):
        with mock.patch.object(fwd, "_run_wsl",
                               return_value=_completed("exited")):
            fwd._control_action("stop")
            self.assertTrue(fwd._operator_stopped)
        with mock.patch.object(fwd, "_run_wsl",
                               return_value=_completed("running")):
            fwd._control_action("start")
            self.assertFalse(fwd._operator_stopped)


class ProcessLaneControlTests(RelayCase):
    """A plain-process upstream: stop by PID, start only with a command."""

    def setUp(self):
        super().setUp()
        fwd._control_token = "tok"
        # A process upstream the scan attributed to pid 4242. Nothing listens
        # there; the control path never touches the upstream itself.
        fwd._upstream = 41999
        fwd._port_owner[41999] = "4242"

    def _post(self, action: str) -> bytes:
        line = (f"POST /__control?token=tok&action={action} "
                f"HTTP/1.1\r\n\r\n").encode()
        return raw_request(self.port, [line])

    def test_stop_terminates_the_pid_and_latches(self):
        seen = []
        with mock.patch.object(fwd, "_run_host",
                               side_effect=lambda a, timeout=10.0:
                               seen.append(a) or _completed()):
            out = self._post("stop")
        self.assertTrue(out.startswith(b"HTTP/1.1 200"), out[:60])
        self.assertEqual(_body(out)["status"], "stopped")
        self.assertEqual(len(seen), 1)
        self.assertIn("4242", seen[0])
        self.assertTrue(fwd._operator_stopped)

    def test_start_without_command_is_409(self):
        out = self._post("start")
        self.assertTrue(out.startswith(b"HTTP/1.1 409"), out[:60])
        self.assertIn(b"upstream-cmd", out)

    def test_start_with_command_spawns_and_clears_latch(self):
        fwd.UPSTREAM_CMD = "llama-server -m model.gguf --port 41999"
        fwd._operator_stopped = True
        spawned = []
        with mock.patch.object(fwd, "_spawn_host",
                               side_effect=lambda a: spawned.append(a)
                               or mock.Mock()):
            out = self._post("start")
        self.assertTrue(out.startswith(b"HTTP/1.1 200"), out[:60])
        self.assertEqual(_body(out)["status"], "starting")
        self.assertEqual(spawned[0][0], "llama-server")
        self.assertIn("41999", spawned[0])
        self.assertFalse(fwd._operator_stopped)

    def test_restart_stops_then_starts(self):
        fwd.UPSTREAM_CMD = "llama-server --port 41999"
        killed, spawned = [], []
        with mock.patch.object(fwd, "_run_host",
                               side_effect=lambda a, timeout=10.0:
                               killed.append(a) or _completed()), \
                mock.patch.object(fwd, "_spawn_host",
                                  side_effect=lambda a: spawned.append(a)
                                  or mock.Mock()):
            out = self._post("restart")
        self.assertEqual(_body(out)["status"], "starting")
        self.assertEqual(len(killed), 1)
        self.assertEqual(len(spawned), 1)

    def test_get_is_405_on_a_controllable_process_lane(self):
        out = raw_request(self.port,
                          [b"GET /__control?token=tok&action=stop HTTP/1.1\r\n\r\n"])
        self.assertTrue(out.startswith(b"HTTP/1.1 405"), out[:60])

    def test_nothing_to_control_is_404(self):
        fwd._port_owner.clear()
        fwd.UPSTREAM_CMD = None
        out = self._post("stop")
        self.assertTrue(out.startswith(b"HTTP/1.1 404"), out[:60])

    def test_response_connection_close(self):
        with mock.patch.object(fwd, "_run_host", return_value=_completed()):
            out = self._post("stop")
        self.assertIn(b"Connection: close", out)


NETSTAT = (
    "  Proto  Local Address          Foreign Address        State           PID\n"
    "  TCP    0.0.0.0:135            0.0.0.0:0              LISTENING       1234\n"
    "  TCP    127.0.0.1:41999        0.0.0.0:0              LISTENING       4242\n"
    "  TCP    127.0.0.1:8890         0.0.0.0:0              LISTENING       777\n"
    "  TCP    127.0.0.1:41999        127.0.0.1:50000        ESTABLISHED     4242\n"
)


class PortPidTests(ForwarderCase):
    """_port_pid: one netstat pass, any executable, one port."""

    def test_finds_the_listening_owner(self):
        with mock.patch.object(fwd, "_run_host", return_value=_completed(NETSTAT)), \
                mock.patch.object(fwd.os, "name", "nt"):
            self.assertEqual(fwd._port_pid(41999), "4242")
            self.assertEqual(fwd._port_pid(8890), "777")

    def test_unknown_port_is_none(self):
        with mock.patch.object(fwd, "_run_host", return_value=_completed(NETSTAT)), \
                mock.patch.object(fwd.os, "name", "nt"):
            self.assertIsNone(fwd._port_pid(5))

    def test_netstat_failure_is_none(self):
        with mock.patch.object(fwd, "_run_host", return_value=None), \
                mock.patch.object(fwd.os, "name", "nt"):
            self.assertIsNone(fwd._port_pid(41999))

    def test_refresh_records_and_forgets(self):
        fwd._upstream = 41999
        with mock.patch.object(fwd, "_run_host", return_value=_completed(NETSTAT)), \
                mock.patch.object(fwd.os, "name", "nt"):
            fwd._refresh_upstream_pid()
            self.assertEqual(fwd._upstream_pid(), "4242")
        with mock.patch.object(fwd, "_run_host", return_value=_completed("")), \
                mock.patch.object(fwd.os, "name", "nt"):
            fwd._refresh_upstream_pid()
            self.assertIsNone(fwd._upstream_pid())


class StopResolvesPidOnDemandTests(RelayCase):
    """A stop right after startup, before the monitor has run: the handler
    resolves the PID itself, then terminates it."""

    def test_stop_with_empty_owner_map(self):
        fwd._control_token = "tok"
        fwd._upstream = 41999
        fwd._port_owner.clear()
        seen = []

        def fake_host(args, timeout=10.0):
            seen.append(args)
            if args[0] == "netstat":
                return _completed(NETSTAT)
            return _completed()

        with mock.patch.object(fwd, "_run_host", side_effect=fake_host), \
                mock.patch.object(fwd.os, "name", "nt"):
            out = raw_request(self.port, [b"POST /__control?token=tok&action=stop HTTP/1.1\r\n\r\n"])
        self.assertTrue(out.startswith(b"HTTP/1.1 200"), out[:80])
        self.assertEqual(_body(out)["status"], "stopped")
        self.assertEqual([a[0] for a in seen], ["netstat", "taskkill"])
        self.assertIn("4242", seen[1])


class SnapshotUnloadFieldsTests(ForwarderCase):

    def test_fields_present_and_typed(self):
        fwd._upstream = 41999
        fwd._port_owner[41999] = "4242"
        fwd._operator_stopped = True
        fwd.UPSTREAM_CMD = "x"
        s = stats.snapshot(fwd, dict(fwd._stats))
        self.assertTrue(s["operator_stopped"])
        self.assertEqual(s["upstream_pid"], "4242")
        self.assertTrue(s["upstream_cmd"])

    def test_container_lane_has_no_pid(self):
        fwd.WSL_DISTRO, fwd.CONTAINER_NAME = "d", "c"
        fwd._upstream = 30000
        fwd._port_owner[30000] = "1"
        s = stats.snapshot(fwd, dict(fwd._stats))
        self.assertIsNone(s["upstream_pid"])
