"""UTF-8 console output for Windows terminals and spawned Python processes."""

import os
import sys


def configure_utf8_output():
    # Environment variables apply to new interpreters, including spawn workers.
    os.environ["PYTHONIOENCODING"] = "utf-8:backslashreplace"
    os.environ["PYTHONUTF8"] = "1"

    # The current interpreter has already opened its standard streams.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="backslashreplace")
