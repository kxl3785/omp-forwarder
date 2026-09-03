"""Discovery: parsing tasklist and netstat, and discover()'s choice.

The shell-outs are replaced with recorded output, so these run on any
platform. Health checks are real, against FakeUpstream."""
from __future__ import annotations

import os
import subprocess
from unittest import mock

from omp_forwarder import forwarder as fwd

from .helpers import ForwarderCase, free_port

TASKLIST = ('"llama-server.exe","39900","Console","1","2,415,044 K"\n'
            '"llama-server.exe","40001","Console","1","1,024 K"\n')
TASKLIST_NONE = ("INFO: No tasks are running which match the specified "
                 "criteria.\n")

NETSTAT = """
Active Connections

  Proto  Local Address          Foreign Address        State           PID
  TCP    0.0.0.0:135            0.0.0.0:0              LISTENING       1234
  TCP    127.0.0.1:8888         0.0.0.0:0              LISTENING       28204
  TCP    127.0.0.1:8788         0.0.0.0:0              LISTENING       39900
  TCP    127.0.0.1:55084        0.0.0.0:0              LISTENING       39900
  TCP    127.0.0.1:55084        127.0.0.1:60123        ESTABLISHED     39900
  TCP    127.0.0.1:60123        127.0.0.1:55084        ESTABLISHED     31308
  TCP    0.0.0.0:55084          0.0.0.0:0              LISTENING       39900
  TCP    127.0.0.1:60008        0.0.0.0:0              LISTENING       40001
  TCP    127.0.0.1:61000        0.0.0.0:0              LISTENING       50000
"""


def _completed(stdout: str) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout,
                                       stderr="")


class ServerPidsTests(ForwarderCase):

    def test_parses_tasklist_csv(self):
        with mock.patch.object(os, "name", "nt"), \
                mock.patch.object(subprocess, "run",
                                  return_value=_completed(TASKLIST)) as run:
            self.assertEqual(fwd._server_pids(), {"39900", "40001"})
        self.assertIn("tasklist", run.call_args[0][0][0])

    def test_no_matching_process_gives_empty_set(self):
        with mock.patch.object(os, "name", "nt"), \
                mock.patch.object(subprocess, "run",
                                  return_value=_completed(TASKLIST_NONE)):
            self.assertEqual(fwd._server_pids(), set())

    def test_tasklist_failure_gives_empty_set(self):
        with mock.patch.object(os, "name", "nt"), \
                mock.patch.object(subprocess, "run",
                                  side_effect=FileNotFoundError("tasklist")):
            self.assertEqual(fwd._server_pids(), set())

    def test_non_windows_does_not_shell_out(self):
        with mock.patch.object(os, "name", "posix"), \
                mock.patch.object(subprocess, "run",
                                  side_effect=AssertionError("ran")):
            self.assertEqual(fwd._server_pids(), set())


class ListeningPortsTests(ForwarderCase):

    def test_listening_ports_of_the_given_pids_highest_first(self):
        with mock.patch.object(subprocess, "run",
                               return_value=_completed(NETSTAT)):
            ports = fwd._listening_ports({"39900", "40001"})
        # 8788 and 55084 belong to 39900, 60008 to 40001. 55084 appears on two
        # addresses and is listed once. 8888 and 61000 are other PIDs, and the
        # ESTABLISHED rows are not listeners.
        self.assertEqual(ports, [60008, 55084, 8788])

    def test_excluded_ports_are_dropped(self):
        fwd.EXCLUDE_PORTS = {8788, 60008}
        with mock.patch.object(subprocess, "run",
                               return_value=_completed(NETSTAT)):
            self.assertEqual(fwd._listening_ports({"39900", "40001"}),
                             [55084])

    def test_no_pids_means_no_netstat(self):
        with mock.patch.object(subprocess, "run",
                               side_effect=AssertionError("ran")):
            self.assertEqual(fwd._listening_ports(set()), [])

    def test_netstat_failure_gives_empty_list(self):
        with mock.patch.object(subprocess, "run",
                               side_effect=subprocess.TimeoutExpired("n", 1)):
            self.assertEqual(fwd._listening_ports({"39900"}), [])


class HealthyTests(ForwarderCase):

    def test_true_only_for_a_200_health_endpoint(self):
        ok = self.fake()
        loading = self.fake(healthy=False)
        self.assertTrue(fwd._healthy(ok.port))
        self.assertFalse(fwd._healthy(loading.port))
        self.assertFalse(fwd._healthy(free_port()))


class DiscoverTests(ForwarderCase):

    def test_forced_port_skips_scanning(self):
        fwd.FORCED_UPSTREAM = 4242
        with mock.patch.object(fwd, "_server_pids",
                               side_effect=AssertionError("scanned")):
            self.assertEqual(fwd.discover(), 4242)
            self.assertEqual(fwd.discover(force=True), 4242)
        self.assertEqual(fwd._upstream, 4242)

    def test_takes_the_first_healthy_port_in_scan_order(self):
        loading = self.fake(healthy=False)
        ok = self.fake()
        fwd._stats["model"] = "old-model"
        with mock.patch.object(fwd, "_server_pids", return_value={"1"}), \
                mock.patch.object(fwd, "_listening_ports",
                                  return_value=[loading.port, ok.port]):
            self.assertEqual(fwd.discover(), ok.port)
        self.assertEqual(fwd._upstream, ok.port)
        # A new upstream means a new model; the cached name must go.
        self.assertEqual(fwd._stats["model"], "")

    def test_no_healthy_port_gives_none(self):
        loading = self.fake(healthy=False)
        fwd._upstream = loading.port
        with mock.patch.object(fwd, "_server_pids", return_value={"1"}), \
                mock.patch.object(fwd, "_listening_ports",
                                  return_value=[loading.port]):
            self.assertIsNone(fwd.discover())
        self.assertIsNone(fwd._upstream)

    def test_healthy_cached_port_is_returned_without_a_scan(self):
        ok = self.fake()
        fwd._upstream = ok.port
        fwd._stats["model"] = "kept"
        with mock.patch.object(fwd, "_server_pids",
                               side_effect=AssertionError("scanned")):
            self.assertEqual(fwd.discover(), ok.port)
        self.assertEqual(fwd._stats["model"], "kept")

    def test_force_rescans_even_when_cached_port_is_healthy(self):
        old = self.fake()
        new = self.fake()
        fwd._upstream = old.port
        with mock.patch.object(fwd, "_server_pids", return_value={"1"}), \
                mock.patch.object(fwd, "_listening_ports",
                                  return_value=[new.port]):
            self.assertEqual(fwd.discover(force=True), new.port)
        self.assertEqual(fwd._upstream, new.port)

    def test_unhealthy_cached_port_triggers_a_scan(self):
        new = self.fake()
        fwd._upstream = free_port()
        with mock.patch.object(fwd, "_server_pids", return_value={"1"}), \
                mock.patch.object(fwd, "_listening_ports",
                                  return_value=[new.port]):
            self.assertEqual(fwd.discover(), new.port)


class ParseArgsTests(ForwarderCase):

    def test_defaults(self):
        a = fwd._parse_args([])
        self.assertEqual((a.port, a.studio_port, a.upstream_port,
                          a.exclude_port, a.tray),
                         (8890, 8888, None, [], False))

    def test_exclude_port_repeats(self):
        a = fwd._parse_args(["--exclude-port", "1", "--exclude-port", "2",
                             "--port", "9000", "--upstream-port", "55084"])
        self.assertEqual(a.exclude_port, [1, 2])
        self.assertEqual((a.port, a.upstream_port), (9000, 55084))
