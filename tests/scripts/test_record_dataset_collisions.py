#!/usr/bin/env python

# Copyright 2026 The XenseRobotics Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""What happens when a recording cannot create its dataset directory.

Both behaviours here were found by a failed record on the bench, and they
compounded: startup died, left an empty directory behind, and every later
attempt hit the same wall — reported as `AttributeError: 'NoneType' object has
no attribute 'push_to_hub'`, which points nowhere near the cause.
"""

import re
from pathlib import Path

import pytest

from lerobot.datasets.lerobot_dataset import LeRobotDataset

FEATURES = {"a": {"dtype": "float32", "shape": (1,), "names": None}}


def test_an_empty_leftover_directory_says_it_is_safe_to_delete(tmp_path):
    """The common case: a run that died during startup blocks every retry."""
    root = tmp_path / "ds"
    root.mkdir()

    with pytest.raises(FileExistsError) as excinfo:
        LeRobotDataset.create("t/x", 30, root=root, features=FEATURES)

    message = str(excinfo.value)
    assert str(root) in message
    assert "empty" in message.lower()
    assert "delete" in message.lower()


def test_a_populated_directory_offers_resume_instead_of_deletion(tmp_path):
    """Telling someone to delete a directory holding real episodes is dangerous."""
    root = tmp_path / "ds"
    root.mkdir()
    (root / "episode.parquet").write_bytes(b"")

    with pytest.raises(FileExistsError) as excinfo:
        LeRobotDataset.create("t/x", 30, root=root, features=FEATURES)

    message = str(excinfo.value)
    assert "resume" in message.lower()
    assert "empty" not in message.lower(), "must not call a populated directory empty"


def test_teardown_does_not_push_a_dataset_that_was_never_created():
    """record()'s finally block runs even when create() raised.

    An unguarded push_to_hub there replaces the real failure with an
    AttributeError on None — which is what made the original report unreadable.
    A source check, because reaching this path needs a full record() run.
    """
    source = Path("src/lerobot/scripts/lerobot_record.py").read_text()

    guard = re.search(
        r"if cfg\.dataset\.push_to_hub and dataset is not None:", source
    )
    assert guard, (
        "push_to_hub in record()'s finally block must be guarded against a "
        "dataset that was never created, the way finalize() above it is"
    )
