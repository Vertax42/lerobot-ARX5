#!/usr/bin/env python

# Copyright 2026 The XenseRobotics Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Pin down how a recipe turns a single arm's gripper cameras on.

The wiring is derived, not written: a recipe sets ``auto_discover_cameras`` on
the ``gripper:`` block, and the robot config turns that into the private flag its
driver branches on. Nothing about that is visible at the point of use, so a
change to the derivation would first show up as a bench where discovery silently
stopped running — the arm connects, the wrist camera is simply absent.

These tests also guard the half that is easy to get wrong by hand: the fisheye
knobs have to survive the trip from YAML, and the derived flag has to stay
*underivable* from YAML. A recipe that could set ``_taccap_autodiscover``
directly would let the two disagree.

The bimanual configs are included in the same parametrization on purpose. The
whole point of this change was that the four arms should behave the same here;
testing only the two new ones would let them drift right back apart.
"""

import dataclasses
import io
from importlib.util import find_spec

import draccus
import pytest

# The Flexiv config imports `flexiv_rt` at module scope, and that SDK is not on
# PyPI — so on a host without it, importing the config to read its *dataclass
# fields* fails. Same guard the rest of tests/robots uses. The Elite config has
# no such import and needs no guard: its cases run everywhere, which is worth
# keeping, since skipping is how coverage quietly disappears.

# (id, config module, config class, vendor SDK needed to import it — or None)
ARMS = [
    (
        "flexiv_rizon4_rt",
        "lerobot.robots.flexiv_rizon4_rt.config_flexiv_rizon4_rt",
        "FlexivRizon4RTConfig",
        "flexiv_rt",
    ),
    ("elite_cs66_rt", "lerobot.robots.elite_cs66_rt.config_elite_cs66_rt", "EliteCS66RTConfig", None),
    (
        "bi_flexiv_rizon4_rt",
        "lerobot.robots.bi_flexiv_rizon4_rt.config_bi_flexiv_rizon4_rt",
        "BiFlexivRizon4RTConfig",
        "flexiv_rt",
    ),
    (
        "bi_elite_cs66_rt",
        "lerobot.robots.bi_elite_cs66_rt.config_bi_elite_cs66_rt",
        "BiEliteCS66RTConfig",
        None,
    ),
]

SINGLE = [a for a in ARMS if not a[0].startswith("bi_")]


def _cls(arm):
    """Import an arm's config class, skipping when its vendor SDK is absent."""
    _, module, name, sdk = arm
    if sdk is not None and find_spec(sdk) is None:
        pytest.skip(f"{sdk} not importable")
    return getattr(__import__(module, fromlist=[name]), name)


def _load(arm, body: str):
    return draccus.load(_cls(arm), io.StringIO(body))


def _gripper(gtype: str = "taccap_follower", *, discover: bool = True, side: str | None = "left") -> str:
    block = f"gripper:\n  type: {gtype}\n  auto_discover_cameras: {str(discover).lower()}\n"
    if side is not None:
        block += f"  side: {side}\n"
    return block


IDS = [a[0] for a in ARMS]
SINGLE_IDS = [a[0] for a in SINGLE]


# --------------------------------------------------------------------------- #
# The fisheye knobs reach the config from a recipe
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("arm", ARMS, ids=IDS)
def test_the_fisheye_knobs_are_settable_from_a_recipe(arm):
    """They are the whole operator-facing surface of this feature. A knob that
    cannot be reached from YAML cannot be turned on at all."""
    cfg = _load(arm, "undistort_wrist_cameras: true\nwrist_fisheye_balance: 0.35\n")

    assert cfg.undistort_wrist_cameras is True
    assert cfg.wrist_fisheye_balance == 0.35


@pytest.mark.parametrize("arm", ARMS, ids=IDS)
def test_rectification_is_off_unless_a_recipe_asks(arm):
    """Nothing changes for a rig that has not opted in."""
    cfg = _load(arm, "{}\n")

    assert cfg.undistort_wrist_cameras is False
    assert cfg.wrist_fisheye_balance == 0.0


@pytest.mark.parametrize("arm", ARMS, ids=IDS)
def test_the_derived_discovery_flags_are_not_settable_from_a_recipe(arm):
    """They are derived from the gripper block. A recipe that could set them
    directly could put them at odds with the block they describe."""
    fields = {f.name: f for f in dataclasses.fields(_cls(arm))}

    for name in ("_taccap_autodiscover", "_serial_autodiscover"):
        assert name in fields, f"{arm[0]} lost {name}"
        assert fields[name].init is False, f"{arm[0]}: {name} must not be settable from a recipe"


# --------------------------------------------------------------------------- #
# The gripper block drives discovery
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("arm", ARMS, ids=IDS)
def test_a_taccap_block_asking_for_discovery_arms_the_taccap_sweep(arm):
    cfg = _load(arm, _gripper("taccap_follower", discover=True, side=None))

    assert cfg._taccap_autodiscover is True
    assert cfg._serial_autodiscover is False


@pytest.mark.parametrize("arm", ARMS, ids=IDS)
def test_a_serial_block_arms_the_serial_sweep_instead(arm):
    """The two sweeps read different buses; arming the wrong one finds nothing."""
    cfg = _load(arm, _gripper("serial", discover=True, side=None))

    assert cfg._serial_autodiscover is True
    assert cfg._taccap_autodiscover is False


@pytest.mark.parametrize("arm", ARMS, ids=IDS)
def test_discovery_turned_off_in_the_block_arms_neither(arm):
    """With it off the recipe pins every camera itself, and a sweep at connect
    would fight what the recipe already said."""
    cfg = _load(arm, _gripper("taccap_follower", discover=False, side=None))

    assert cfg._taccap_autodiscover is False
    assert cfg._serial_autodiscover is False


@pytest.mark.parametrize("arm", ARMS, ids=IDS)
def test_no_gripper_at_all_arms_neither(arm):
    """There is nothing to discover cameras off of."""
    cfg = _load(arm, "{}\n")

    assert cfg._taccap_autodiscover is False
    assert cfg._serial_autodiscover is False


# --------------------------------------------------------------------------- #
# Single arms take their side from the gripper
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("arm", SINGLE, ids=SINGLE_IDS)
@pytest.mark.parametrize("side", ["left", "right"])
def test_a_single_arm_keeps_the_side_its_recipe_pinned(arm, side):
    """A single arm has no side of its own, so the gripper's is what names its
    camera keys. A bimanual recipe must not pin it — a single-arm one must."""
    cfg = _load(arm, _gripper(side=side))

    assert cfg.gripper.side == side


@pytest.mark.parametrize("arm", SINGLE, ids=SINGLE_IDS)
def test_a_single_arm_defaults_to_the_left_gripper(arm):
    """Left is the taccap block's own default; the arm does not override it."""
    cfg = _load(arm, _gripper(side=None))

    assert cfg.gripper.side == "left"


@pytest.mark.parametrize("arm", SINGLE, ids=SINGLE_IDS)
def test_a_single_arm_can_still_wire_its_tactile_sensors(arm):
    """Discovery needs this flag, and it did not exist on the single arms before
    they had discovery at all."""
    assert _load(arm, "{}\n").enable_tactile_sensors is True
    assert _load(arm, "enable_tactile_sensors: false\n").enable_tactile_sensors is False
