"""Test package. Puts src/ on sys.path so `python -m unittest` works from a
clone with nothing installed, the same way run_forwarder.bat does."""
import pathlib
import sys

_SRC = str(pathlib.Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
