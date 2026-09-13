"""The NInfer adapter: the dashboard's only source for such a lane.

NInfer answers no /metrics, no /slots and no /props, so a lane running it was
blank in flight. These tests pin the reader against recorded records in the
engine's own schema-v10 shape.
"""
import json
import time
import unittest

from omp_forwarder import forwarder as fwd
from omp_forwarder import stats

from .helpers import ForwarderCase


def _start(**engine):
    eng = {"tp": 1, "devices": [0], "max_context": 131072,
           "effective_max_context": 262144, "kv_capacity": 232768,
           "speculative_backend": "mtp", "speculative_draft_window": 3}
    eng.update(engine)
    return {"event": "server_start", "timestamp_unix_ms": 1000,
            "engine": eng, "server": {"default_thinking": False},
            "artifact": {"path": "/models/qwen3_8_27b_nvfp4.ninfer",
                         "target": "qwen3_8_27b"}}


def _tput(ms, running=0, waiting=0, decode=0.0, prefill=0.0):
    return {"event": "throughput", "timestamp_unix_ms": ms,
            "scheduler": {"running": running, "waiting": waiting,
                          "prefilling": 0, "decode_ready": 0},
            "throughput_tokens_per_second": {"decode": decode,
                                             "prefill": prefill}}


def _done(ms, rid, prompt, cached, out, decode_s, drafted=0, accepted=0):
    return {"event": "request_done", "timestamp_unix_ms": ms,
            "request": {"request_id": rid},
            "result": {"prompt_tokens": prompt, "completion_tokens": out,
                       "prefix_cache_hit_tokens": cached,
                       "finish_reason": "stop_token"},
            "timings_seconds": {"prefill": 1.0, "decode": decode_s},
            "speculative": {"drafted_tokens": drafted,
                            "accepted_tokens": accepted}}


def _blob(events):
    return "\n".join(json.dumps(e) for e in events) + "\n"


class NinferLogTests(unittest.TestCase):
    def _read(self, events):
        return lambda path, n: _blob(events)

    def test_facts_come_from_the_boot_record(self):
        now = 10.0
        d = stats.ninfer_log_stats("x", now=now, reader=self._read([_start()]))
        self.assertEqual(d["facts"], {
            "engine": "ninfer", "thinking": "off", "speculative": "mtp 3",
            "parallel": "tp=1",
            "model_path": "/models/qwen3_8_27b_nvfp4.ninfer"})
        # The per-sequence ceiling, not the pool, and the pool reported apart.
        self.assertEqual(d["ctx"], 131072)
        self.assertEqual(d["kv_capacity"], 232768)

    def test_two_devices_read_as_tp2(self):
        d = stats.ninfer_log_stats(
            "x", now=10.0, reader=self._read([_start(tp=2, devices=[0, 1])]))
        self.assertEqual(d["facts"]["parallel"], "tp=2")

    def test_in_flight_state_comes_from_the_newest_throughput_record(self):
        now = 1000.0
        events = [_tput(int((now - 20) * 1000), running=1, decode=10.0),
                  _tput(int((now - 2) * 1000), running=3, waiting=1,
                        decode=180.5, prefill=4200.0)]
        d = stats.ninfer_log_stats("x", now=now, reader=self._read(events))
        self.assertEqual(d["scheduler"]["running"], 3)
        self.assertEqual(d["scheduler"]["waiting"], 1)
        # The scheduler counts are the engine's and are trusted. The rate in
        # that same record is not: see the decode-rate tests below.
        self.assertEqual(d["scheduler"]["prefilling"], 0)

    def test_an_undiluted_window_sets_the_decode_rate(self):
        # No prefill happened in this window, so its average IS the stream.
        now = 1000.0
        d = stats.ninfer_log_stats(
            "x", now=now,
            reader=self._read([_tput(int((now - 2) * 1000), running=1,
                                     decode=206.4, prefill=0.0)]))
        self.assertEqual(d["rates"]["decode"], 206.4)

    def test_a_window_that_also_prefilled_is_not_trusted(self):
        """The engine averages over the whole interval, prefill included.

        Measured 2026-09-13 over an hour of real traffic: windows carrying a
        prefill read 44.0 tok/s while the requests finishing inside them ran
        at 184.9, and 570 of 588 windows carried one. Reporting that number
        is the llama-server prefill bug in another engine's clothes."""
        now = 1000.0
        d = stats.ninfer_log_stats(
            "x", now=now,
            reader=self._read([_tput(int((now - 2) * 1000), running=1,
                                     decode=44.0, prefill=4200.0)]))
        self.assertEqual(d["rates"]["decode"], 0.0)
        # Prefill is measured over prefill work and needs no such care.
        self.assertEqual(d["rates"]["prefill"], 4200.0)

    def test_the_live_rate_comes_from_decode_time(self):
        # 200 tokens over 1.0 s and 400 over 2.0 s is 199.3 tok/s over decode
        # time, whatever the engine's diluted window says.
        now = 1000.0
        events = [_done(int((now - 10) * 1000), 1, 500, 400, 200, 1.0),
                  _done(int((now - 5) * 1000), 2, 500, 400, 400, 2.0),
                  _tput(int((now - 2) * 1000), running=1,
                        decode=44.0, prefill=4200.0)]
        d = stats.ninfer_log_stats("x", now=now, reader=self._read(events))
        self.assertAlmostEqual(d["rates"]["decode"], 598 / 3.0, places=3)

    def test_a_request_outside_the_window_does_not_set_the_rate(self):
        now = 1000.0
        old = now - stats.NINFER_LIVE_WINDOW_S - 30
        events = [_done(int(old * 1000), 1, 500, 400, 200, 1.0),
                  _tput(int((now - 2) * 1000), running=1,
                        decode=44.0, prefill=4200.0)]
        d = stats.ninfer_log_stats("x", now=now, reader=self._read(events))
        self.assertEqual(d["rates"]["decode"], 0.0)

    def test_a_stale_record_reports_idle_rather_than_an_old_number(self):
        """A forwarder that outlives its engine must not show last hour's rate."""
        now = 1000.0
        d = stats.ninfer_log_stats(
            "x", now=now,
            reader=self._read([_tput(int((now - 600) * 1000), running=4,
                                     decode=200.0)]))
        self.assertEqual(d["rates"]["decode"], 0.0)
        self.assertEqual(d["scheduler"]["running"], 0)
        self.assertGreater(d["scheduler"]["age_s"], stats.NINFER_STALE_S)

    def test_finished_requests_become_the_recent_list_newest_first(self):
        now = 500.0
        events = [_done(100_000, 1, 2000, 0, 51, 0.5),
                  _done(200_000, 2, 90_000, 88_000, 101, 1.0)]
        d = stats.ninfer_log_stats("x", now=now, reader=self._read(events))
        self.assertEqual([r["slot"] for r in d["recent"]], ["req 2", "req 1"])
        first = d["recent"][0]
        self.assertEqual((first["prompt"], first["cached"], first["tokens"]),
                         (90_000, 88_000, 101))
        self.assertEqual(first["rate"], 100.0)     # (101 - 1) / 1.0

    def test_totals_carry_the_cache_and_acceptance_evidence(self):
        d = stats.ninfer_log_stats(
            "x", now=10.0,
            reader=self._read([_done(1000, 1, 100, 90, 11, 1.0,
                                     drafted=10, accepted=8)]))
        self.assertAlmostEqual(d["totals"]["cache_hit_rate"], 0.9)
        self.assertAlmostEqual(d["totals"]["acceptance"], 0.8)

    def test_no_path_and_no_file_yield_nothing_rather_than_zeros(self):
        self.assertEqual(stats.ninfer_log_stats(None), {})
        self.assertEqual(stats.ninfer_log_stats("x", reader=lambda p, n: ""), {})
        self.assertEqual(
            stats.ninfer_log_stats(r"C:\no\such\ninfer\log.jsonl"), {})


class NinferSnapshotTests(ForwarderCase):
    """The snapshot must carry what the page renders: a lane row, the recent
    list and the Deployment facts."""

    def setUp(self):
        super().setUp()
        fwd.LISTEN_PORT = 8890
        fwd._upstream = 49600
        fwd._upstream_healthy = True
        now = time.time()
        fwd._ninfer_stats = {
            "facts": {"engine": "ninfer", "thinking": "off",
                      "speculative": "mtp 3", "parallel": "tp=1",
                      "model_path": "/models/x.ninfer"},
            "ctx": 131072, "kv_capacity": 232768,
            "scheduler": {"running": 2, "waiting": 1, "prefilling": 0,
                          "age_s": 1.0},
            "rates": {"decode": 176.0, "prefill": 4200.0},
            "totals": {"prompt": 100, "cached": 93, "completion": 10,
                       "cache_hit_rate": 0.93, "acceptance": 0.78},
            "recent": [{"slot": "req 7", "tokens": 911, "seconds": 5.2,
                        "rate": 176.0, "prompt": 92636, "cached": 17184,
                        "ended": now}],
        }

    def test_lane_row_reports_the_engine_in_flight(self):
        d = stats.snapshot(fwd, dict(fwd._stats))
        self.assertEqual(len(d["lane_rows"]), 1)
        row = d["lane_rows"][0]
        self.assertEqual(row["engine"], "ninfer")
        self.assertEqual((row["running"], row["queued"]), (2, 1))
        self.assertEqual(row["rate"], 176.0)
        self.assertEqual(row["ctx"], 131072)

    def test_recent_list_is_tagged_with_the_lane(self):
        d = stats.snapshot(fwd, dict(fwd._stats))
        self.assertEqual(len(d["recent_streams"]), 1)
        self.assertEqual(d["recent_streams"][0]["lane"], 8890)
        self.assertEqual(d["recent_streams"][0]["rate"], 176.0)

    def test_facts_beat_the_http_probe_that_cannot_see_this_engine(self):
        fwd._upstream_facts = {"engine": "unknown", "thinking": "unknown"}
        d = stats.snapshot(fwd, dict(fwd._stats))
        self.assertEqual(d["facts"]["engine"], "ninfer")
        self.assertEqual(d["lane_facts"][0]["facts"]["engine"], "ninfer")

    def test_a_lane_without_the_log_keeps_the_probe_facts(self):
        fwd._ninfer_stats = {}
        fwd._upstream_facts = {"engine": "llama-server"}
        d = stats.snapshot(fwd, dict(fwd._stats))
        self.assertEqual(d["facts"]["engine"], "llama-server")
        self.assertEqual(d["lane_rows"], [])


class NinferPresetLogTests(ForwarderCase):
    def test_preset_log_path_fills_gpu_port_and_container_name(self):
        p = {"port": "4960{gpu}", "container": "ninfer{gpu}",
             "log": "/root/ninfer/logs/{name}.requests.jsonl"}
        self.assertEqual(fwd._preset_log(p, 1),
                         "/root/ninfer/logs/ninfer1.requests.jsonl")

    def test_a_preset_without_a_log_yields_none(self):
        self.assertIsNone(fwd._preset_log({"port": "3000{gpu}"}, 0))


class NinferTokenTallyTests(ForwarderCase):
    """The Tokens card on an NInfer lane.

    The engine publishes no token counters, so the forwarder folds its log's
    finished requests into the same tally llama-server's /metrics feeds. The
    card read zero through a day of real traffic before this existed."""

    def _stats_for(self, events):
        return stats.ninfer_log_stats("x", now=10.0,
                                      reader=lambda path, n: _blob(events))

    def test_a_finished_request_moves_the_tally(self):
        nin = self._stats_for([_start(), _done(2000, 1, 500, 200, 60, 1.0)])
        fwd._tally_ninfer_tokens(nin)
        # prompt_tokens includes the cached prefix; the two must stay apart.
        self.assertEqual(fwd._stats["tok_prompt"], 300)
        self.assertEqual(fwd._stats["tok_cached"], 200)
        self.assertEqual(fwd._stats["tok_gen"], 60)

    def test_the_overlapping_tail_is_not_counted_twice(self):
        first = [_start(), _done(2000, 1, 500, 200, 60, 1.0)]
        fwd._tally_ninfer_tokens(self._stats_for(first))
        second = first + [_done(3000, 2, 100, 0, 40, 1.0)]
        fwd._tally_ninfer_tokens(self._stats_for(second))
        self.assertEqual(fwd._stats["tok_prompt"], 400)
        self.assertEqual(fwd._stats["tok_gen"], 100)

    def test_a_restarted_engine_repeats_its_request_ids(self):
        # Request 1 again, at a later timestamp: a different request.
        fwd._tally_ninfer_tokens(self._stats_for([_done(2000, 1, 100, 0, 10, 1.0)]))
        fwd._tally_ninfer_tokens(self._stats_for([_done(9000, 1, 100, 0, 10, 1.0)]))
        self.assertEqual(fwd._stats["tok_gen"], 20)

    def test_the_tail_that_was_already_there_is_a_baseline(self):
        old = [_start(), _done(2000, 1, 500, 200, 60, 1.0)]
        fwd._tally_ninfer_tokens(self._stats_for(old), baseline=True)
        self.assertEqual(fwd._stats["tok_gen"], 0)
        fwd._tally_ninfer_tokens(self._stats_for(old + [_done(3000, 2, 100, 0, 40, 1.0)]))
        self.assertEqual(fwd._stats["tok_gen"], 40)

    def test_an_empty_log_changes_nothing(self):
        fwd._tally_ninfer_tokens({})
        self.assertEqual(fwd._stats["tok_prompt"], 0)

    def test_the_counted_set_stays_bounded(self):
        fwd.NINFER_COUNTED_MAX = 5
        self.addCleanup(setattr, fwd, "NINFER_COUNTED_MAX", 4000)
        for i in range(20):
            fwd._tally_ninfer_tokens(self._stats_for([_done(1000 + i, i, 10, 0, 1, 1.0)]))
        self.assertLessEqual(len(fwd._ninfer_counted), 5)
        self.assertEqual(fwd._stats["tok_gen"], 20)


if __name__ == "__main__":
    unittest.main()


class NinferStreamCountTests(ForwarderCase):
    """How many requests the engine decodes at once, and where it shows.

    It is a launch argument, fixed for the life of the process, and no engine
    here answers for it over HTTP. A lane serving one at a time looks
    identical to a lane serving four until the page says which."""

    def _read(self, events):
        return lambda path, n: "\n".join(json.dumps(e) for e in events)

    def test_the_stream_count_comes_from_the_boot_record(self):
        d = stats.ninfer_log_stats(
            "x", now=10.0, reader=self._read([_start(max_concurrency=2)]))
        self.assertEqual(d["streams"], 2)

    def test_an_engine_that_does_not_say_reports_none(self):
        d = stats.ninfer_log_stats("x", now=10.0, reader=self._read([_start()]))
        self.assertEqual(d["streams"], 0)

    def test_the_lane_row_and_the_in_flight_card_show_it(self):
        self.assertIn("stream", stats.PAGE)
        self.assertIn("NIN.streams", stats.PAGE)
        self.assertIn("l.streams", stats.PAGE)

    def test_the_fleet_adds_the_lanes_up(self):
        own = {"listen": 8890, "ninfer": {"running": 1, "streams": 2}}
        peer = {"listen": 8891, "ninfer": {"running": 0, "streams": 2}}
        m = stats.merge_snapshots(own, [peer])
        self.assertEqual(m["ninfer"]["streams"], 4)
