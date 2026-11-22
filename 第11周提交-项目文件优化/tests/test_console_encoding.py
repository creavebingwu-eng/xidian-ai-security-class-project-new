import os
from pathlib import Path
import subprocess
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ConsoleEncodingTests(unittest.TestCase):
    def test_gbk_process_and_child_can_print_unicode(self):
        # Reproduce the original Windows encoding even on a UTF-8 machine.
        env = dict(os.environ, PYTHONIOENCODING="gbk:strict", PYTHONUTF8="0")
        script = """
import subprocess
import sys
assert sys.stdout.encoding.lower() == 'gbk'
from ai_scientist.console_encoding import configure_utf8_output
configure_utf8_output()
from rich.console import Console
text = '\\u2713 \\u4e2d\\u6587 \\u2019'
Console(legacy_windows=True, force_terminal=True, color_system=None).print(text)
print(text, file=sys.stderr)
child = subprocess.run(
    [sys.executable, '-c', 'import sys; print(' + repr(text) + '); print(' + repr(text) + ', file=sys.stderr)'],
    capture_output=True, check=True,
)
assert child.stdout.decode('utf-8').strip() == text
assert child.stderr.decode('utf-8').strip() == text
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", errors="replace"))
        expected = "\u2713 \u4e2d\u6587 \u2019"
        self.assertIn(expected, result.stdout.decode("utf-8"))
        self.assertIn(expected, result.stderr.decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
