"""The sidebar restart stops workers before it clears the RH latch."""
from __future__ import annotations

import ast
import os
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path

_DASH = Path(__file__).resolve().parents[1] / "scripts" / "oi_dashboard_app.py"
_KEEP_ASSIGN = {"_SERVICE_SPECS"}
_KEEP_FN = {"_restart_bundle_script"}

_NS: dict = {"shlex": shlex, "os": os, "Path": Path}
_tree = ast.parse(_DASH.read_text())
for _node in _tree.body:
    names = []
    if isinstance(_node, ast.Assign):
        names = [t.id for t in _node.targets if isinstance(t, ast.Name)]
    elif isinstance(_node, ast.AnnAssign) and isinstance(_node.target, ast.Name):
        names = [_node.target.id]
    if any(n in _KEEP_ASSIGN for n in names):
        exec(compile(ast.Module([_node], type_ignores=[]), str(_DASH), "exec"), _NS)
    elif isinstance(_node, ast.FunctionDef) and _node.name in _KEEP_FN:
        exec(compile(ast.Module([_node], type_ignores=[]), str(_DASH), "exec"), _NS)


class _FakePopen:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs


class RestartBundleTests(unittest.TestCase):
    def test_stops_poller_before_clearing_the_latch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            popped: list = []

            def popen(*args, **kwargs):
                popped.append((args, kwargs))
                return _FakePopen(*args, **kwargs)

            # The extracted function looks up REPO_ROOT in the namespace
            # it was exec'd into, not in the caller's locals.
            _NS["REPO_ROOT"] = root
            _NS["subprocess"] = type(
                "S",
                (),
                {"Popen": staticmethod(popen), "DEVNULL": subprocess.DEVNULL},
            )
            _NS["_restart_bundle_script"](
                service_keys=["price_poller"],
                tickers=["SPY"],
                days_ahead=5,
                force_mfa=False,
            )
            text = (root / "logs" / "dashboard-restart.sh").read_text()

        self.assertIn("price_poller.py", text)
        self.assertIn("--stop", text)
        self.assertIn("clear_manual_login_required()", text)
        self.assertIn("clear_login_cooldown()", text)
        self.assertNotIn("rm -f", text)
        stop_at = text.index("--stop")
        clear_at = text.index("clear_manual_login_required()")
        start_at = text.index("--replace")
        self.assertLess(stop_at, clear_at)
        self.assertLess(clear_at, start_at)
        self.assertEqual(len(popped), 1)
        self.assertFalse(any(a == "--stop" for a in popped[0][0]))


if __name__ == "__main__":
    unittest.main()
