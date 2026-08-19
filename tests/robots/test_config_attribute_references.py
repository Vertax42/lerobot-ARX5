#!/usr/bin/env python

# Copyright 2026 The XenseRobotics Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Catch `self.config.<gone>` after a config field is renamed or removed.

Robot configs are wide, flat dataclasses and their drivers read dozens of fields.
Deleting one is easy; finding every read of it is not, and Python does not
complain until the line actually executes — which for a driver usually means
`connect()`, i.e. only with hardware attached, mid-session.

That is not hypothetical: folding the flat gripper knobs into a nested `gripper:`
block left twelve reads of `config.gripper_init_open` and friends behind, and the
first anyone knew was an AttributeError after both arms had already powered up.

This walks each robot package's AST for attribute reads on `self.config` (and on
the `config` parameter of `__init__`) and asserts the name is a real field. It is
static, so it needs no hardware and runs in milliseconds.
"""

import ast
import dataclasses
import importlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# (package dir, config module, config class). Robots whose SDK may be missing are
# skipped rather than failed — the import is what needs the SDK, not the check.
ROBOTS = [
    ("bi_flexiv_rizon4_rt", "config_bi_flexiv_rizon4_rt", "BiFlexivRizon4RTConfig"),
    ("bi_elite_cs66_rt", "config_bi_elite_cs66_rt", "BiEliteCS66RTConfig"),
    ("flexiv_rizon4_rt", "config_flexiv_rizon4_rt", "FlexivRizon4RTConfig"),
    ("elite_cs66_rt", "config_elite_cs66_rt", "EliteCS66RTConfig"),
]


def _config_class(pkg: str, mod: str, cls_name: str):
    try:
        module = importlib.import_module(f"lerobot.robots.{pkg}.{mod}")
    except Exception as exc:  # SDK not built on this host
        pytest.skip(f"{pkg}: config not importable ({type(exc).__name__})")
    return getattr(module, cls_name)


class _ConfigReads(ast.NodeVisitor):
    """Collect `<base>.<attr>` reads where <base> is self.config or a local config.

    Only ``self.config`` and a bare ``config`` are treated as the robot's config.
    ``gripper.config`` and the like are deliberately not: they belong to some other
    object whose fields this test knows nothing about.
    """

    def __init__(self):
        self.reads: list[tuple[str, int]] = []

    def visit_Attribute(self, node: ast.Attribute) -> None:
        value = node.value
        is_self_config = (
            isinstance(value, ast.Attribute)
            and value.attr == "config"
            and isinstance(value.value, ast.Name)
            and value.value.id == "self"
        )
        is_bare_config = isinstance(value, ast.Name) and value.id == "config"
        if is_self_config or is_bare_config:
            self.reads.append((node.attr, node.lineno))
        self.generic_visit(node)


@pytest.mark.parametrize(("pkg", "mod", "cls_name"), ROBOTS, ids=[r[0] for r in ROBOTS])
def test_driver_only_reads_fields_the_config_has(pkg, mod, cls_name):
    cls = _config_class(pkg, mod, cls_name)
    # dir() as well as fields(): properties and ClassVars are legitimate reads too.
    known = {f.name for f in dataclasses.fields(cls)} | set(dir(cls))

    bad: list[str] = []
    for py in sorted((REPO_ROOT / "src" / "lerobot" / "robots" / pkg).glob("*.py")):
        visitor = _ConfigReads()
        visitor.visit(ast.parse(py.read_text()))
        for attr, lineno in visitor.reads:
            if attr not in known:
                bad.append(f"{py.name}:{lineno} config.{attr}")

    assert not bad, (
        f"{cls_name} has no such field(s) — these reads would raise AttributeError "
        f"the moment the line runs (often only once hardware is attached):\n  " + "\n  ".join(bad)
    )
