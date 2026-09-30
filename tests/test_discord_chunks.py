"""Discord replies must stay within the 2000-character content limit."""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src" / "julia" / "discorder.py"
_KEEP_ASSIGN = {"_DISCORD_MAX_LEN", "HELP_TEXT"}
_KEEP_FN = {"_discord_chunks"}

_NS: dict = {}
_tree = ast.parse(_SRC.read_text())
for _node in _tree.body:
    names = []
    if isinstance(_node, ast.Assign):
        names = [t.id for t in _node.targets if isinstance(t, ast.Name)]
    if any(n in _KEEP_ASSIGN for n in names):
        exec(compile(ast.Module([_node], type_ignores=[]), str(_SRC), "exec"), _NS)
    elif isinstance(_node, ast.FunctionDef) and _node.name in _KEEP_FN:
        exec(compile(ast.Module([_node], type_ignores=[]), str(_SRC), "exec"), _NS)

_discord_chunks = _NS["_discord_chunks"]
HELP_TEXT = _NS["HELP_TEXT"]
_DISCORD_MAX_LEN = _NS["_DISCORD_MAX_LEN"]


def _fence_balanced(chunk: str) -> bool:
    marks = sum(1 for line in chunk.splitlines() if line.strip().startswith("```"))
    return marks % 2 == 0


class ChunkTests(unittest.TestCase):
    def test_short_text_is_one_message(self) -> None:
        self.assertEqual(_discord_chunks("hello"), ["hello"])
        self.assertEqual(_discord_chunks(""), [""])

    def test_help_fits_discord_limit(self) -> None:
        # The live failure: !lia / !lia help sent this in one message.
        self.assertGreater(len(HELP_TEXT), _DISCORD_MAX_LEN)
        chunks = _discord_chunks(HELP_TEXT)
        self.assertGreaterEqual(len(chunks), 2)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), _DISCORD_MAX_LEN)
            self.assertTrue(_fence_balanced(chunk), chunk[:80])
        joined = "\n".join(chunks)
        for line in (
            "!lia help",
            "!lia open spread",
            "!lia close spread",
            "!lia watch SPY 759",
            "Buy/sell place **real orders**.",
        ):
            self.assertIn(line, joined)

    def test_splits_on_newlines(self) -> None:
        text = "\n".join(f"line {i:02d} ...." for i in range(30))
        chunks = _discord_chunks(text, limit=40)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 40)
        self.assertEqual("\n".join(chunks), text)

    def test_open_fence_is_closed_and_reopened(self) -> None:
        lines = ["```", *[f"row {i}" for i in range(20)], "```"]
        text = "\n".join(lines)
        chunks = _discord_chunks(text, limit=30)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 30)
            self.assertTrue(_fence_balanced(chunk), chunk)
        body = "\n".join(chunks)
        for i in range(20):
            self.assertIn(f"row {i}", body)

    def test_oversized_line_is_sliced(self) -> None:
        chunks = _discord_chunks("x" * 50, limit=20)
        self.assertEqual(chunks, ["x" * 20, "x" * 20, "x" * 10])
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 20)


if __name__ == "__main__":
    unittest.main()
