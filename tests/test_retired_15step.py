"""Guards against reintroducing the retired short-horizon production contract."""

from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_PRODUCTION_PATTERNS = (
    re.compile(r"history8_current7_future15"),
    re.compile(r"future_steps\s*[=:]\s*15\b"),
    re.compile(r"[\"']future_steps[\"']\s*:\s*15\b"),
    re.compile(r"trajectory_steps[^\n\d]*15\b"),
    re.compile(r"sample_count[^\n\d]*23\b"),
    re.compile(r"MOTION_SNIPPET_(?:FUTURE_STEPS|SAMPLE_COUNT)\s*=\s*(?:15|23)\b"),
    re.compile(r"\(15,\s*3\)"),
)


def test_production_tree_has_no_retired_15_step_contract() -> None:
    matches: list[str] = []
    for directory in ("scripts", "src", "configs"):
        for path in sorted((ROOT / directory).rglob("*")):
            if not path.is_file() or path.suffix not in {".py", ".yaml", ".json"}:
                continue
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if any(pattern.search(line) for pattern in FORBIDDEN_PRODUCTION_PATTERNS):
                    matches.append(
                        f"{path.relative_to(ROOT)}:{line_number}: {line.strip()}"
                    )
    assert matches == []
