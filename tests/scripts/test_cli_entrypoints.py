#!/usr/bin/env python

# Copyright 2026 The XenseRobotics Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Check the CLI entry points still parse arguments.

``@parser.wrap()`` is what turns these functions into CLI commands. Every test in
this repo calls them directly with a config object, so losing the decorator
breaks nothing a test would notice — the whole suite stays green while
``lerobot-teleoperate --robot.type=...`` stops accepting arguments.

It went missing once, cut along with the code above it during a refactor.
"""

import pytest

from lerobot.scripts import lerobot_record, lerobot_replay, lerobot_teleoperate


@pytest.mark.parametrize(
    ("module", "func_name"),
    [
        (lerobot_teleoperate, "teleoperate"),
        (lerobot_record, "record"),
        (lerobot_replay, "replay"),
    ],
)
def test_entrypoint_is_wrapped_for_the_cli(module, func_name):
    func = getattr(module, func_name)
    # parser.wrap() replaces the function, so the original is on __wrapped__.
    assert hasattr(func, "__wrapped__"), (
        f"{module.__name__}.{func_name} lost its @parser.wrap() decorator — "
        "the CLI would stop parsing arguments"
    )
