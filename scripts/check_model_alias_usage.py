#!/usr/bin/env python3
"""Fail if parser modules hardcode provider model IDs instead of aliases."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARSERS_DIR = ROOT / "researcher_ai" / "parsers"

# Explicit provider model ids we disallow in parser code.
DISALLOWED = re.compile(r"['\"](?:gpt-[^'\"]+|claude-[^'\"]+|gemini-[^'\"]+)['\"]")
ALLOW_COMMENT = "model-alias-check: ignore"


def main() -> int:
    violations: list[str] = []
    for path in sorted(PARSERS_DIR.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), start=1):
            if ALLOW_COMMENT in line:
                continue
            for m in DISALLOWED.finditer(line):
                token = m.group(0)
                violations.append(f"{path.relative_to(ROOT)}:{i}: hardcoded model id {token}")

    if violations:
        print("Model alias guard failed. Use logical aliases from models.yaml in parser modules.")
        for v in violations:
            print(v)
        return 1

    print("Model alias guard passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
