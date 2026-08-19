#!/usr/bin/env python

# Copyright 2026 The XenseRobotics Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Keep logging calls inside what this project's logger actually implements.

``get_logger()`` returns a spdlog ``SinkLogger``, not a stdlib ``logging.Logger``.
It has ``.warn`` but no ``.warning``, no ``.exception``, no ``.fatal`` — so the
habitual stdlib spelling raises AttributeError instead of logging anything.

Nothing catches that in review or in a normal run, because these calls sit on
paths that only execute when something has already gone wrong. Seven of them
were live in this repo at once, including the record loop's dropped-frame
warning (which fires exactly when a recording is degrading) and two inside
``except`` blocks, where the AttributeError would replace the error being
reported.
"""

import re
from pathlib import Path

import pytest

from lerobot.utils.robot_utils import get_logger

SRC = Path(__file__).resolve().parents[1] / "src" / "lerobot"

#: stdlib Logger methods a SinkLogger does not implement, that read as natural.
NOT_ON_SINKLOGGER = ("warning", "exception", "fatal", "setLevel", "isEnabledFor")


def _logger_names(text: str) -> set[str]:
    """Names bound to a get_logger() result in this file."""
    return set(re.findall(r"(\w+(?:\.\w+)?)\s*=\s*get_logger\(", text))


def _offences() -> list[str]:
    out = []
    for path in sorted(SRC.rglob("*.py")):
        if "__pycache__" in str(path):
            continue
        text = path.read_text(errors="ignore")
        if "get_logger" not in text:
            continue
        names = _logger_names(text)
        if not names:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            for name in names:
                for method in NOT_ON_SINKLOGGER:
                    if re.search(rf"\b{re.escape(name)}\.{method}\s*\(", line):
                        rel = path.relative_to(SRC.parents[1])
                        out.append(f"{rel}:{lineno}: .{method}() on a spdlog logger")
    return out


@pytest.mark.parametrize("method", NOT_ON_SINKLOGGER)
def test_the_logger_really_lacks_these(method):
    """Guard the guard: if SinkLogger grows one, this list should shrink."""
    assert not hasattr(get_logger("api-probe"), method), (
        f"SinkLogger now has .{method}() — drop it from NOT_ON_SINKLOGGER"
    )


def test_warn_is_the_spelling_that_works():
    assert hasattr(get_logger("api-probe"), "warn")


def test_no_source_file_calls_a_method_the_logger_lacks():
    offences = _offences()
    assert not offences, (
        "these would raise AttributeError instead of logging:\n  "
        + "\n  ".join(offences)
        + "\nUse .warn() — get_logger() returns a spdlog SinkLogger."
    )
