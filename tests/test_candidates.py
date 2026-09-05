"""Candidate-port discovery.

A model server in a Docker container inside WSL2 (SGLang on 30000, say)
has no Windows executable behind its listener, so netstat matching by
executable path cannot see it. Discovery therefore accepts candidate
ports on the side: --candidate-port entries, and the container's
published port derived from `docker port`. Both are probed with /health
like everything else; --prefer picks the kind when both are healthy.

The shell-outs are replaced with recorded output, so these run on any
platform. Health checks are real, against FakeUpstream."""
from __future__ import annotations

import os
import subprocess
from unittest import mock

from omp_forwarder import forwarder as fwd

from .helpers import ForwarderCase, RelayCase, free_port, raw_request

class DockerPortTests(ForwarderCase):
    """`docker port` output parsing, via the _run_wsl seam."""

    def _docker_responder(self, stdout: str):
        def fake_run_wsl(args, timeout=10.0):
            if "docker" in args and "port" in args:
                return subprocess.CompletedProcess(
                    args=args, returncode=0, stdout=stdout, stderr="")
            raise AssertionError(f"unexpected _run_wsl call: {args}")
        return fake_run_wsl

    def test_single_published_port(self):
        fwd.WSL_DISTRO = "Ubuntu-24.04"
        fwd.CONTAINER_NAME = "sgl"
        with mock.patch.object(fwd, "_run_wsl",
                               side_effect=self._docker_responder(
                                   "30000/tcp -> 0.0.0.0:30000\n")):
            self.assertEqual(fwd._container_port_from_docker(), 30000)
        self.assertEqual(fwd._container_port, 30000)

    def test_ipv4_ipv6_and_blank_lines_dedupe_to_one(self):
        # The IPv6 line repeats the same host port; a blank line must not
        # matter. The answer is the single host port, deduped.
        out = ("30000/tcp -> 0.0.0.0:30000\n"
               "30000/tcp -> [::]:30000\n"
               "\n")
        fwd.WSL_DISTRO = "Ubuntu-24.04"
        fwd.CONTAINER_NAME = "sgl"
        with mock.patch.object(fwd, "_run_wsl",
                               side_effect=self._docker_responder(out)):
            self.assertEqual(fwd._container_port_from_docker(), 30000)

    def test_failure_gives_no_candidates(self):
        # `docker port` fails: there is simply no derived candidate this
        # pass. The last good value is kept; with none, None.
        fwd.WSL_DISTRO = "Ubuntu-24.04"
        fwd.CONTAINER_NAME = "sgl"
        with mock.patch.object(fwd, "_run_wsl", return_value=None):
            self.assertIsNone(fwd._container_port_from_docker())
        self.assertIsNone(fwd._container_port)

    def test_container_mode_off_derives_nothing(self):
        with mock.patch.object(fwd, "_run_wsl",
                               side_effect=AssertionError("ran")):
            self.assertIsNone(fwd._container_port_from_docker())
        self.assertIsNone(fwd._container_port)

    def test_parse_container_port_direct(self):
        self.assertEqual(
            fwd._parse_container_port("30000/tcp -> 0.0.0.0:30000\n"
                                      "30000/tcp -> [::]:30000\n\n"),
            30000)
        # Two different published ports is ambiguous: no single answer.
        self.assertIsNone(
            fwd._parse_container_port("30000/tcp -> 0.0.0.0:30000\n"
                                      "30001/tcp -> 0.0.0.0:30001\n"))
        # No port at the end of a line: malformed, no candidate.
        self.assertIsNone(fwd._parse_container_port("some status line\n"))
        self.assertIsNone(fwd._parse_container_port(""))


class CandidateDiscoverTests(ForwarderCase):
    """discover() with candidate ports, on the recorded fixtures."""

    def _no_llama_server(self):
        return (mock.patch.object(fwd, "_server_pids", return_value=set()),
                mock.patch.object(fwd, "_listening_ports",
                                  side_effect=lambda pids: [] if not pids
                                  else [55084]))

    def test_candidate_chosen_when_no_llama_server(self):
        # Scenario 1: nothing in the fixtures matches a llama-server, one
        # healthy candidate port -> it is chosen, and its kind is
        # "candidate".
        cand = self.fake()
        fwd.CANDIDATE_PORTS = [cand.port]
        a, b = self._no_llama_server()
        with a, b:
            self.assertEqual(fwd.discover(force=True), cand.port)
        self.assertEqual(fwd._upstream, cand.port)
        self.assertEqual(fwd._upstream_kind, "candidate")

    def test_default_prefer_picks_llama_server_over_candidate(self):
        # Scenario 2: both healthy, default preference -> the
        # executable-matched llama-server wins.
        server = self.fake()
        cand = self.fake()
        fwd.CANDIDATE_PORTS = [cand.port]
        fwd._port_owner[server.port] = "39900"
        with mock.patch.object(fwd, "_server_pids", return_value={"39900"}), \
                mock.patch.object(fwd, "_listening_ports",
                                  return_value=[server.port]):
            self.assertEqual(fwd.discover(force=True), server.port)
        self.assertEqual(fwd._upstream_kind, "llama-server")

    def test_prefer_candidate_picks_candidate(self):
        # Scenario 3: same, but --prefer candidate -> the candidate wins.
        server = self.fake()
        cand = self.fake()
        fwd.CANDIDATE_PORTS = [cand.port]
        fwd.PREFER = "candidate"
        fwd._port_owner[server.port] = "39900"
        with mock.patch.object(fwd, "_server_pids", return_value={"39900"}), \
                mock.patch.object(fwd, "_listening_ports",
                                  return_value=[server.port]):
            self.assertEqual(fwd.discover(force=True), cand.port)
        self.assertEqual(fwd._upstream_kind, "candidate")

    def test_first_unhealthy_candidate_is_skipped(self):
        # Scenario 4: two candidate ports, the first unhealthy -> the
        # second is chosen.
        first = self.fake(healthy=False)
        second = self.fake()
        fwd.CANDIDATE_PORTS = [first.port, second.port]
        a, b = self._no_llama_server()
        with a, b:
            self.assertEqual(fwd.discover(force=True), second.port)
        self.assertEqual(fwd._upstream_kind, "candidate")

    def test_flag_order_precedes_derived_container_port(self):
        # A flag given before the derived port is tried first, even when
        # both are healthy: flags are the user's explicit order.
        flagged = self.fake()
        fwd.CANDIDATE_PORTS = [flagged.port]
        fwd._container_port = free_port()     # a derived port that is dead
        a, b = self._no_llama_server()
        with a, b:
            self.assertEqual(fwd.discover(force=True), flagged.port)
        self.assertEqual(fwd._upstream_kind, "candidate")

    def test_derived_container_port_becomes_a_candidate(self):
        # The monitor thread's derived port is picked up by discovery as a
        # candidate when the llama-server scan finds nothing.
        derived = self.fake()
        fwd._container_port = derived.port
        a, b = self._no_llama_server()
        with a, b:
            self.assertEqual(fwd.discover(force=True), derived.port)
        self.assertEqual(fwd._upstream_kind, "candidate")

    def test_nothing_healthy_gives_none(self):
        # Scenario 5: no healthy candidate and no healthy llama-server.
        # The caller (relay) then takes the existing 503/Retry-After path,
        # so discover() must come back None.
        first = self.fake(healthy=False)
        fwd.CANDIDATE_PORTS = [first.port]
        a, b = self._no_llama_server()
        with a, b:
            self.assertIsNone(fwd.discover(force=True))
        self.assertIsNone(fwd._upstream)
        self.assertIsNone(fwd._upstream_kind)

    def test_forced_upstream_ignores_candidates_and_prefer(self):
        # Scenario 7: --upstream-port still wins over everything; the kind
        # reads "explicit", and neither the scan nor a probe runs.
        cand = self.fake()
        fwd.FORCED_UPSTREAM = cand.port
        fwd.CANDIDATE_PORTS = [free_port()]
        fwd.PREFER = "candidate"
        with mock.patch.object(fwd, "_server_pids",
                               side_effect=AssertionError("scanned")), \
                mock.patch.object(fwd, "_listening_ports",
                                  side_effect=AssertionError("scanned")), \
                mock.patch.object(fwd, "_healthy",
                                  side_effect=AssertionError("probed")):
            self.assertEqual(fwd.discover(), cand.port)
        self.assertEqual(fwd._upstream_kind, "explicit")

    def test_excluded_port_never_becomes_a_candidate(self):
        # Studio's port (and our own listen port) are in EXCLUDE_PORTS and
        # must be dropped even if passed as --candidate-port: treating
        # Studio as a candidate without --studio-fallback is the 401 bug.
        studio = free_port()
        fwd.EXCLUDE_PORTS = {studio}
        self.assertNotIn(studio, fwd._candidate_ports())

    def test_candidates_are_deduped_and_excluded(self):
        a = free_port()
        fwd.CANDIDATE_PORTS = [a, a]
        self.assertEqual(fwd._candidate_ports(), [a])


class StudioPortNeverCandidateTests(RelayCase):
    """Scenario 8: Studio's port is never chosen from candidates.

    If --candidate-port is pointed at Studio's own port and nothing else is
    healthy, the relay must answer its own 503 -- it must not relay to
    Studio, which would hand the client a 401. This is the same 401
    protection the original no-model test asserts; extended so a
    candidate list cannot reintroduce it."""

    def test_candidate_pointed_at_studio_is_excluded(self):
        studio = self.fake(lambda m, p, r: [b"HTTP/1.1 401 Unauthorized\r\n"
                                             b"Content-Length: 0\r\n\r\n"])
        fwd.STUDIO_PORT = studio.port
        # EXCLUDE_PORTS is reset in setUp; point the candidate list straight
        # at Studio's port so only that candidate would otherwise win.
        fwd.CANDIDATE_PORTS = [studio.port]
        fwd.EXCLUDE_PORTS = {self.port, fwd.STUDIO_PORT}
        out = raw_request(self.port, [b"GET /v1/models HTTP/1.1\r\n\r\n"])
        self.assertTrue(out.startswith(b"HTTP/1.1 503"))
        self.assertIn(b"Retry-After: 5", out)
        # Studio was never touched, so it cannot contribute a 401.
        self.assertEqual(studio.received, [])
        self.assertEqual(fwd._stats["unavailable"], 1)
        self.assertEqual(fwd._stats["fallbacks"], 0)


class CandidateParseArgsTests(ForwarderCase):

    def test_candidate_port_repeats_in_order(self):
        a = fwd._parse_args(["--candidate-port", "30000",
                             "--candidate-port", "30001"])
        self.assertEqual(a.candidate_port, [30000, 30001])
        self.assertEqual(a.prefer, "llama-server")

    def test_prefer_defaults_to_llama_server(self):
        a = fwd._parse_args([])
        self.assertEqual(a.prefer, "llama-server")

    def test_prefer_candidate(self):
        a = fwd._parse_args(["--prefer", "candidate",
                             "--candidate-port", "30000"])
        self.assertEqual((a.prefer, a.candidate_port),
                         ("candidate", [30000]))

    def test_defaults_leave_candidate_list_empty(self):
        a = fwd._parse_args([])
        self.assertEqual(a.candidate_port, [])
