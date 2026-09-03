"""_note_status: the deliberately naive HTTP status counter.

It must count a read that begins with a status line, and must never count
anything else -- a body chunk, a split status line, a malformed one. It may
undercount; it must not miscount."""
from __future__ import annotations

from omp_forwarder import forwarder as fwd

from .helpers import ForwarderCase


def counts() -> tuple[int, int, int]:
    return fwd._stats["2xx"], fwd._stats["4xx"], fwd._stats["5xx"]


class NoteStatusTests(ForwarderCase):

    def test_buckets_by_hundreds(self):
        fwd._note_status(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")
        fwd._note_status(b"HTTP/1.1 404 Not Found\r\n\r\n")
        fwd._note_status(b"HTTP/1.1 500 Internal Server Error\r\n\r\n")
        fwd._note_status(b"HTTP/1.1 503 Service Unavailable\r\n\r\n")
        self.assertEqual(counts(), (1, 1, 2))

    def test_http_1_0_counts_too(self):
        fwd._note_status(b"HTTP/1.0 200 OK\r\n\r\n")
        self.assertEqual(counts(), (1, 0, 0))

    def test_ignores_body_chunks(self):
        fwd._note_status(b'data: {"choices":[]}\n\n')
        fwd._note_status(b"")
        fwd._note_status(b"\r\n")
        self.assertEqual(counts(), (0, 0, 0))

    def test_ignores_a_status_line_split_across_reads(self):
        # Undercount, never miscount: neither half is a status line.
        fwd._note_status(b"HTTP/1.")
        fwd._note_status(b"1 500 Internal Server Error\r\n\r\n")
        self.assertEqual(counts(), (0, 0, 0))

    def test_ignores_a_status_line_with_no_numeric_code(self):
        fwd._note_status(b"HTTP/1.1 OK\r\n\r\n")
        fwd._note_status(b"HTTP/1.1\r\n")
        self.assertEqual(counts(), (0, 0, 0))

    def test_does_not_count_a_status_line_in_the_middle_of_a_read(self):
        # Pipelined replies can put a second status line mid-read. The
        # counter only looks at the head, so that one is not counted.
        fwd._note_status(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n"
                         b"HTTP/1.1 500 Oops\r\n\r\n")
        self.assertEqual(counts(), (1, 0, 0))
