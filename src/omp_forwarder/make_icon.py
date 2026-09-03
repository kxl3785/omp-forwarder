"""Generate assets/omp-forwarder.ico -- a relay arrow, drawn from scratch.

    python -m omp_forwarder.make_icon

Pure standard library: math and struct only, no Pillow or cairo. The shipped
.ico is committed, so you only need this if you want to change the mark.

TWO THINGS WORTH KNOWING if you adapt this.

Every frame is a classic uncompressed BMP entry (BITMAPINFOHEADER, bottom-up
BGRA, plus a 1-bit AND mask), NOT a PNG-in-ICO frame. The Windows shell
decodes PNG frames fine, but GDI+ (System.Drawing.Icon, Icon.ToBitmap) renders
them as per-pixel-alpha noise when the .ico is loaded in-process, so anything
that reads the file through .NET gets garbage. BMP frames load correctly
everywhere, Tk's iconbitmap included.

There is ONE drawing at every size. The weights scale with the frame but the
geometry never changes, so a 16px tray icon is the same logo as the 256px one.
Swapping in a simplified mark below some threshold makes a taskbar button and
the shortcut beside it read as two different logos.
"""
from __future__ import annotations

import math
import os
import struct

_SIZES = (16, 24, 32, 48, 64, 128, 256)

# Cool slate ground, teal glyph. Legible on both light and dark taskbars.
_BG = (0x0f / 255, 0x1e / 255, 0x26 / 255)
_ARROW = (0x5e / 255, 0xd6 / 255, 0xcb / 255)


def _clamp01(v: float) -> float:
    return 0.0 if v < 0 else (1.0 if v > 1 else v)


def _seg_dist(px, py, ax, ay, bx, by) -> float:
    """Distance from a point to a line segment -- used to stroke lines with
    antialiasing, since we have no drawing library."""
    vx, vy = bx - ax, by - ay
    l2 = vx * vx + vy * vy
    t = 0.0 if l2 == 0 else (((px - ax) * vx + (py - ay) * vy) / l2)
    t = 0.0 if t < 0 else (1.0 if t > 1 else t)
    return math.hypot(px - (ax + t * vx), py - (ay + t * vy))


def _over(dst, src, sa):
    """`src` (rgb) at straight alpha `sa` composited over `dst` (rgba)."""
    dr, dg, db, da = dst
    a = sa + da * (1 - sa)
    if a <= 0:
        return (0.0, 0.0, 0.0, 0.0)
    inv = da * (1 - sa)
    return ((src[0] * sa + dr * inv) / a, (src[1] * sa + dg * inv) / a,
            (src[2] * sa + db * inv) / a, a)


def _rrect_cover(px, py, cx, cy, ex, rr) -> float:
    """Antialiased coverage of a rounded square, as a signed-distance field."""
    qx = max(abs(px - cx) - (ex - rr), 0.0)
    qy = max(abs(py - cy) - (ex - rr), 0.0)
    return _clamp01(0.5 - (math.hypot(qx, qy) - rr))


def _arrow_strokes(s: int, cy: float):
    """Shaft plus two barbs, proportional to the frame."""
    tipx = s * 0.755
    head = s * 0.175
    return [(s * 0.235, cy, tipx, cy),
            (s * 0.555, cy - head, tipx, cy),
            (s * 0.555, cy + head, tipx, cy)]


def render(s: int) -> bytes:
    """RGBA bytes (straight alpha) for one s x s frame."""
    cx = cy = s / 2.0
    ex = s / 2.0 - s * 0.055
    rr = s * 0.24
    hw = max(0.95, s * 0.058)       # stroke half-width; must hold at 16px
    strokes = _arrow_strokes(s, cy)
    buf = bytearray(s * s * 4)
    for py in range(s):
        yc = py + 0.5
        for px in range(s):
            xc = px + 0.5
            d = (0.0, 0.0, 0.0, 0.0)
            ground = _rrect_cover(xc, yc, cx, cy, ex, rr)
            if ground > 0:
                d = _over(d, _BG, ground)
                best = min(_seg_dist(xc, yc, *seg) for seg in strokes)
                a = _clamp01(0.5 - (best - hw))
                if a > 0:
                    # Clip the glyph to the ground so the arrow can never
                    # bleed past the rounded corner.
                    d = _over(d, _ARROW, a * ground)
            i = (py * s + px) * 4
            buf[i] = int(d[0] * 255 + 0.5)
            buf[i + 1] = int(d[1] * 255 + 0.5)
            buf[i + 2] = int(d[2] * 255 + 0.5)
            buf[i + 3] = int(d[3] * 255 + 0.5)
    return bytes(buf)


def bmp_frame(s: int, rgba: bytes) -> bytes:
    """One ICO frame: BITMAPINFOHEADER, bottom-up 32-bit BGRA, 1-bit AND mask
    with rows padded to 32 bits. No compression -- see the module docstring."""
    row = s * 4
    xor = bytearray()
    for y in range(s - 1, -1, -1):                       # bottom-up
        base = y * row
        for x in range(s):
            i = base + x * 4
            xor += bytes((rgba[i + 2], rgba[i + 1], rgba[i], rgba[i + 3]))
    mask_row = ((s + 31) // 32) * 4                      # 1bpp, DWORD-padded
    mask = bytearray()
    for y in range(s - 1, -1, -1):
        bits = bytearray(mask_row)
        base = y * row
        for x in range(s):
            if rgba[base + x * 4 + 3] == 0:              # transparent -> 1
                bits[x >> 3] |= 0x80 >> (x & 7)
        mask += bits
    header = struct.pack("<IiiHHIIiiII", 40, s, s * 2, 1, 32, 0,
                         len(xor) + len(mask), 0, 0, 0, 0)
    return header + bytes(xor) + bytes(mask)


def build_ico() -> bytes:
    images = [(s, bmp_frame(s, render(s))) for s in _SIZES]
    out = struct.pack("<HHH", 0, 1, len(images))
    offset = 6 + 16 * len(images)
    directory, blob = b"", b""
    for s, bmp in images:
        wb = s if s < 256 else 0                         # 256 stored as 0
        directory += struct.pack("<BBBBHHII", wb, wb, 0, 0, 1, 32,
                                 len(bmp), offset)
        blob += bmp
        offset += len(bmp)
    return out + directory + blob


def preview(s: int = 24) -> str:
    """ASCII rendering of one frame, so you can check legibility without
    opening the file."""
    rgba = render(s)
    ramp = " .:-=+*#%@"
    rows = []
    for y in range(s):
        line = ""
        for x in range(s):
            i = (y * s + x) * 4
            r, g, b, a = rgba[i], rgba[i + 1], rgba[i + 2], rgba[i + 3]
            if a < 40:
                line += " "
            else:
                lum = (0.3 * r + 0.59 * g + 0.11 * b) / 255
                line += ramp[min(9, int(lum * 9.99))]
        rows.append(line)
    return "\n".join(rows)


def main() -> None:
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    assets = os.path.join(root, "assets")
    os.makedirs(assets, exist_ok=True)
    out = os.path.join(assets, "omp-forwarder.ico")
    data = build_ico()
    with open(out, "wb") as fh:
        fh.write(data)
    print(f"wrote {out} ({len(data)} bytes, {len(_SIZES)} sizes)")
    print()
    print(preview(24))


if __name__ == "__main__":
    main()
