"""Windows tray icon for the forwarder.

Its own module so the forwarder stays importable and runnable on a box
without pywin32 -- forwarder.py imports this lazily, only for --tray.
Uses Shell_NotifyIcon through win32gui; pystray and Pillow are not required.

Left-click opens the stats page, right-click opens the menu. That is the
Windows convention: the primary action on a plain click, options on a right
click.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading

import win32api
import win32con
import win32gui

_ID_STATUS = 1023
_ID_REDISCOVER = 1024
_ID_LOG = 1025
_ID_EXIT = 1026
_ID_STATS = 1027
_ID_USAGE = 1028
_MSG_TRAY = win32con.WM_USER + 20


def _icon_path() -> str:
    """assets/ sits beside the package root, so walk up out of src/."""
    here = os.path.dirname(os.path.abspath(__file__))
    for up in (2, 3):
        cand = os.path.join(*([here] + [".."] * up + ["assets",
                                                      "omp-forwarder.ico"]))
        cand = os.path.normpath(cand)
        if os.path.exists(cand):
            return cand
    return os.path.normpath(os.path.join(here, "..", "..", "assets",
                                         "omp-forwarder.ico"))


class Tray:
    def __init__(self, fwd, log_path: str | None = None) -> None:
        """fwd is the omp_forwarder module: we read its state, never copy it."""
        self.fwd = fwd
        self.log_path = log_path
        self.hwnd = None
        self._stop = threading.Event()

    # ---- tooltip reflects live state, so hovering answers "is it working?"
    def _tokens(self) -> str:
        """The forwarder's own tally since it started: what went to the local
        model instead of a paid one. See forwarder._tally_tokens."""
        st = getattr(self.fwd, "_stats", {})
        return (f"{st.get('tok_prompt', 0):,} prompt / "
                f"{st.get('tok_gen', 0):,} generated tokens")

    def _tip(self) -> str:
        up = getattr(self.fwd, "_upstream", None)
        where = f"-> 127.0.0.1:{up}" if up else "-> Studio :8888 (no direct server)"
        return (f"omp forwarder :{self.fwd.LISTEN_PORT}  {where}"
                f"\n{self._tokens()}"
                "\nClick for stats")[:127]

    def stats_url(self) -> str:
        return f"http://127.0.0.1:{self.fwd.LISTEN_PORT}/__stats"

    def usage_url(self) -> str:
        return f"http://127.0.0.1:{self.fwd.LISTEN_PORT}/__usage"

    def _open(self, url: str) -> None:
        """startfile, not webbrowser: the latter can spawn a console window
        under pythonw."""
        try:
            os.startfile(url)                       # noqa: S606 - user action
        except Exception:
            pass

    def open_stats(self) -> None:
        self._open(self.stats_url())

    def open_usage(self) -> None:
        self._open(self.usage_url())

    def _add(self) -> None:
        try:
            hicon = win32gui.LoadImage(0, _icon_path(), win32con.IMAGE_ICON,
                                       0, 0, win32con.LR_LOADFROMFILE)
        except Exception:
            hicon = win32gui.LoadIcon(0, win32con.IDI_APPLICATION)
        win32gui.Shell_NotifyIcon(win32gui.NIM_ADD, (
            self.hwnd, 0,
            win32gui.NIF_ICON | win32gui.NIF_MESSAGE | win32gui.NIF_TIP,
            _MSG_TRAY, hicon, self._tip()))

    def refresh(self) -> None:
        if self.hwnd is None:
            return
        try:
            win32gui.Shell_NotifyIcon(win32gui.NIM_MODIFY, (
                self.hwnd, 0, win32gui.NIF_TIP, _MSG_TRAY, 0, self._tip()))
        except Exception:
            pass

    def _menu(self) -> None:
        up = getattr(self.fwd, "_upstream", None)
        menu = win32gui.CreatePopupMenu()
        win32gui.AppendMenu(menu, win32con.MF_STRING | win32con.MF_GRAYED,
                            _ID_STATUS,
                            f"upstream: {up if up else 'none (using Studio)'}")
        win32gui.AppendMenu(menu, win32con.MF_STRING | win32con.MF_GRAYED,
                            _ID_STATUS, self._tokens())
        win32gui.AppendMenu(menu, win32con.MF_SEPARATOR, 0, "")
        # Bold, and first: it is what a left-click does, so the menu should
        # say so rather than hide the primary action among the maintenance ones.
        win32gui.AppendMenu(menu, win32con.MF_STRING | win32con.MF_DEFAULT,
                            _ID_STATS, "Open stats")
        win32gui.AppendMenu(menu, win32con.MF_STRING, _ID_USAGE, "Open usage")
        win32gui.AppendMenu(menu, win32con.MF_STRING, _ID_REDISCOVER,
                            "Re-discover llama-server")
        if self.log_path:
            win32gui.AppendMenu(menu, win32con.MF_STRING, _ID_LOG, "Open log")
        win32gui.AppendMenu(menu, win32con.MF_SEPARATOR, 0, "")
        win32gui.AppendMenu(menu, win32con.MF_STRING, _ID_EXIT, "Exit")
        x, y = win32gui.GetCursorPos()
        win32gui.SetForegroundWindow(self.hwnd)
        win32gui.TrackPopupMenu(menu, win32con.TPM_LEFTALIGN, x, y, 0,
                                self.hwnd, None)
        win32gui.PostMessage(self.hwnd, win32con.WM_NULL, 0, 0)
        win32gui.DestroyMenu(menu)

    def _on_command(self, cid: int) -> None:
        if cid == _ID_STATS:
            self.open_stats()
        elif cid == _ID_USAGE:
            self.open_usage()
        elif cid == _ID_REDISCOVER:
            self.fwd.discover(force=True)
            self.refresh()
        elif cid == _ID_LOG and self.log_path:
            try:
                os.startfile(self.log_path)          # noqa: S606 - user action
            except Exception:
                pass
        elif cid == _ID_EXIT:
            self.stop()

    def _wndproc(self, hwnd, msg, wparam, lparam):
        if msg == _MSG_TRAY:
            # Windows convention: left-click does the primary thing, right
            # click offers the menu. The primary thing here is the stats page.
            if lparam in (win32con.WM_LBUTTONUP, win32con.WM_LBUTTONDBLCLK):
                self.open_stats()
            elif lparam == win32con.WM_RBUTTONUP:
                self._menu()
        elif msg == win32con.WM_COMMAND:
            self._on_command(win32api.LOWORD(wparam))
        elif msg == win32con.WM_DESTROY:
            try:
                win32gui.Shell_NotifyIcon(win32gui.NIM_DELETE, (self.hwnd, 0))
            except Exception:
                pass
            win32gui.PostQuitMessage(0)
        else:
            return win32gui.DefWindowProc(hwnd, msg, wparam, lparam)
        return 0

    def stop(self) -> None:
        self._stop.set()
        if self.hwnd:
            win32gui.PostMessage(self.hwnd, win32con.WM_CLOSE, 0, 0)

    def run(self) -> None:
        """Owns the thread it is called on: Win32 requires the message pump to
        live on the thread that created the window."""
        cls = win32gui.WNDCLASS()
        cls.hInstance = win32api.GetModuleHandle(None)
        cls.lpszClassName = "OmpForwarderTray"
        cls.lpfnWndProc = self._wndproc
        atom = win32gui.RegisterClass(cls)
        self.hwnd = win32gui.CreateWindow(atom, "omp forwarder", 0,
                                          0, 0, 0, 0, 0, 0, cls.hInstance, None)
        win32gui.UpdateWindow(self.hwnd)
        self._add()
        # Keep the tooltip honest without polling the network: discover() is
        # cached, so this is a cheap health touch.
        def tick() -> None:
            while not self._stop.wait(20):
                self.fwd.discover()
                self.refresh()
        threading.Thread(target=tick, daemon=True).start()
        win32gui.PumpMessages()
