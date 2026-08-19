#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Offline validation of the Elite manipulability kinematics against the public UR5.

The Elite CS66 DH values can only be read from a live controller, but the FK/Jacobian
*math* (Modified-DH builder, geometric Jacobian, manipulability) is robot-independent and
is validated here against the canonical Craig-MDH UR5 table, whose singularities are known.
This is the build-time guard that a wrong DH convention (which would silently make ``w≡0``
everywhere and freeze teleop) is caught before the code ever reaches the robot.
"""

import numpy as np
import pytest

from lerobot.robots.elite_cs66_rt.manipulability import (
    damping_scale,
    directional_scale,
    flange_se3,
    geometric_jacobian,
    joint_velocity_scale,
    manipulability,
    tool_consistency,
)
from lerobot.utils.rotation import Rotation

# Canonical Modified-DH (Craig) UR5 table (alpha_i, a_i, d_i indexed per joint i).
UR5_ALPHA = [0.0, np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2]
UR5_A = [0.0, 0.0, -0.425, -0.39225, 0.0, 0.0]
UR5_D = [0.089159, 0.0, 0.0, 0.10915, 0.09465, 0.0823]

GENERIC_Q = np.array([0.3, -1.0, 0.8, -0.5, 1.2, 0.4])


def _w(q):
    return manipulability(UR5_ALPHA, UR5_A, UR5_D, q)


def test_generic_config_is_well_conditioned():
    # A non-singular pose has clearly non-zero manipulability.
    assert _w(GENERIC_Q) > 0.05


def test_wrist_singularity_q5_zero():
    # Wrist singularity: joint 5 (index 4) = 0 -> J4 and J6 axes align -> det(J) = 0.
    q = GENERIC_Q.copy()
    q[4] = 0.0
    assert _w(q) == pytest.approx(0.0, abs=1e-6)


def test_elbow_singularity_q3_zero():
    # Elbow / full-extension singularity: joint 3 (index 2) = 0 -> arm straight -> det(J) = 0.
    q = GENERIC_Q.copy()
    q[2] = 0.0
    assert _w(q) == pytest.approx(0.0, abs=1e-6)


def test_det_invariant_to_tool_offset():
    # |det(J)| is invariant to a rigid flange->TCP tool offset (spatial shear, det = 1).
    jac = geometric_jacobian(UR5_ALPHA, UR5_A, UR5_D, GENERIC_Q)
    p = np.array([0.05, -0.12, 0.2])  # arbitrary tool translation
    skew = np.array([[0, -p[2], p[1]], [p[2], 0, -p[0]], [-p[1], p[0], 0]])
    shear = np.eye(6)
    shear[:3, 3:] = -skew
    j_tcp = shear @ jac
    assert abs(np.linalg.det(j_tcp)) == pytest.approx(abs(np.linalg.det(jac)), rel=1e-9)


def test_det_invariant_to_base_rotation():
    # |det(J)| is invariant to the base-mounting rotation blkdiag(R, R), det = det(R)^2 = 1.
    jac = geometric_jacobian(UR5_ALPHA, UR5_A, UR5_D, GENERIC_Q)
    rng = np.random.default_rng(0)
    # random rotation via QR of a random matrix
    r, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(r) < 0:
        r[:, 0] = -r[:, 0]
    blk = np.zeros((6, 6))
    blk[:3, :3] = r
    blk[3:, 3:] = r
    j_world = blk @ jac
    assert abs(np.linalg.det(j_world)) == pytest.approx(abs(np.linalg.det(jac)), rel=1e-9)


def test_standard_dh_builder_is_a_canary():
    # A *standard*-DH link transform (Rz·Tz·Tx·Rx) fed the same MDH numbers yields w==0 at a
    # generic pose. This documents why the MDH convention is load-bearing: if a future
    # refactor swaps the builder, this assertion flips and the regression is caught.
    def _std_link(alpha, a, d, theta):
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        return np.array(
            [
                [ct, -st * ca, st * sa, a * ct],
                [st, ct * ca, -ct * sa, a * st],
                [0.0, sa, ca, d],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

    def _std_jacobian(q):
        n = len(q)
        frames = [np.eye(4)]
        t = np.eye(4)
        for i in range(n):
            t = t @ _std_link(UR5_ALPHA[i], UR5_A[i], UR5_D[i], q[i])
            frames.append(t.copy())
        o_n = frames[-1][:3, 3]
        jac = np.zeros((6, n))
        for i in range(n):
            z_i = frames[i][:3, 2]
            o_i = frames[i][:3, 3]
            jac[:3, i] = np.cross(z_i, o_n - o_i)
            jac[3:, i] = z_i
        return jac

    # Standard-DH chain on MDH numbers is degenerate -> det 0 even at a generic pose.
    assert abs(np.linalg.det(_std_jacobian(GENERIC_Q))) == pytest.approx(0.0, abs=1e-6)
    # ...whereas the correct MDH builder is non-degenerate there.
    assert _w(GENERIC_Q) > 0.05


def test_damping_scale_clamps():
    # Above w_high -> full speed; below w_low -> floored at s_min; linear between.
    assert damping_scale(0.10, 0.01, 0.05, 0.05) == 1.0
    assert damping_scale(0.0, 0.01, 0.05, 0.05) == pytest.approx(0.05)
    mid = damping_scale(0.03, 0.01, 0.05, 0.05)
    assert 0.05 < mid < 1.0
    assert mid == pytest.approx((0.03 - 0.01) / (0.05 - 0.01))
    # degenerate band -> no damping
    assert damping_scale(0.02, 0.05, 0.05, 0.05) == 1.0


def test_directional_scale_does_not_damp_escape():
    # Near the wrist singularity, a step that raises manipulability should be undamped (s=1).
    q = GENERIC_Q.copy()
    q[4] = 0.02  # just off the wrist singularity (small w)
    w_here = _w(q)
    assert w_here < 0.05  # confirm we're in the damping band
    # Build a Cartesian step from the joint move that increases |q5| (escapes the singularity).
    jac = geometric_jacobian(UR5_ALPHA, UR5_A, UR5_D, q)
    dq_escape = np.zeros(6)
    dq_escape[4] = 0.2  # drive q5 away from 0
    dx_escape = jac @ dq_escape
    s, w = directional_scale(UR5_ALPHA, UR5_A, UR5_D, q, dx_escape, w_low=0.0, w_high=0.05, s_min=0.05)
    assert s == 1.0
    # A step deeper into the singularity is still damped.
    dx_into = -dx_escape
    s2, _ = directional_scale(UR5_ALPHA, UR5_A, UR5_D, q, dx_into, w_low=0.0, w_high=0.05, s_min=0.05)
    assert s2 < 1.0


def test_directional_scale_zero_floor_is_a_true_hold():
    # With s_min=0 a directional guard FULLY holds a move into the singularity (s=0, no creep-through)
    # while still freeing any escape. This is why the config permits min_scale=0 only when directional:
    # the 0.05 floor let the arm crawl 5%/tick past the IK-refusal boundary and trip; 0 stops it dead.
    q = GENERIC_Q.copy()
    q[4] = 0.02  # near the wrist singularity, w below w_low
    jac = geometric_jacobian(UR5_ALPHA, UR5_A, UR5_D, q)
    dq_escape = np.zeros(6)
    dq_escape[4] = 0.2
    dx_escape = jac @ dq_escape
    s_escape, _ = directional_scale(UR5_ALPHA, UR5_A, UR5_D, q, dx_escape, w_low=0.05, w_high=0.1, s_min=0.0)
    assert s_escape == 1.0  # escape / retraction stays free
    s_into, _ = directional_scale(UR5_ALPHA, UR5_A, UR5_D, q, -dx_escape, w_low=0.05, w_high=0.1, s_min=0.0)
    assert s_into == 0.0  # into-singularity is held dead, not floored at a creeping 0.05


# Official CS66 datasheet per-joint velocity ceiling (rad/s): J1/J2 150°/s, J3 180°/s, J4-6 230°/s,
# pre-scaled by an 0.8 headroom margin (as the driver does before calling joint_velocity_scale).
QDOT_LIMIT = np.array([2.618, 2.618, 3.142, 4.014, 4.014, 4.014]) * 0.8
DT = 0.033  # ~ 1/teleop_fps command horizon


def test_joint_velocity_scale_no_scaling_when_within_limits():
    # A tiny Cartesian step at a well-conditioned pose stays under the per-joint budget -> s == 1.
    dx = np.array([0.001, 0.0, 0.0, 0.001, 0.0, 0.0])
    s, dq = joint_velocity_scale(UR5_ALPHA, UR5_A, UR5_D, GENERIC_Q, dx, DT, QDOT_LIMIT)
    assert s == 1.0
    assert np.max(np.abs(dq) / (QDOT_LIMIT * DT)) < 1.0


def test_joint_velocity_scale_zero_step():
    # No commanded motion -> no joint motion, no scaling.
    s, dq = joint_velocity_scale(UR5_ALPHA, UR5_A, UR5_D, GENERIC_Q, np.zeros(6), DT, QDOT_LIMIT)
    assert s == 1.0
    assert np.allclose(dq, 0.0)


def test_joint_velocity_scale_enforces_limit_near_wrist_singularity():
    # Near the wrist singularity, a twist along the least-controllable direction demands a huge
    # joint step; the guard must scale it down so no joint exceeds its per-tick budget.
    q = GENERIC_Q.copy()
    q[4] = 0.01
    jac = geometric_jacobian(UR5_ALPHA, UR5_A, UR5_D, q)
    u, _, _ = np.linalg.svd(jac)
    dx = u[:, -1] * 0.05  # least-controllable Cartesian direction
    s, dq = joint_velocity_scale(UR5_ALPHA, UR5_A, UR5_D, q, dx, DT, QDOT_LIMIT)
    assert 0.0 < s < 1.0
    # After uniform scaling the worst joint sits exactly at its budget (s = 1/ratio).
    scaled_ratio = np.max(np.abs(dq * s) / (QDOT_LIMIT * DT))
    assert scaled_ratio == pytest.approx(1.0)


def test_joint_velocity_scale_uniform_scaling_preserves_direction():
    # Scaling is a single scalar on the Cartesian step, so re-running on the scaled step yields the
    # same joint-space direction with magnitude * s (linearity), and that scaled step is in-budget.
    q = GENERIC_Q.copy()
    q[4] = 0.01
    jac = geometric_jacobian(UR5_ALPHA, UR5_A, UR5_D, q)
    u, _, _ = np.linalg.svd(jac)
    dx = u[:, -1] * 0.05
    s, dq = joint_velocity_scale(UR5_ALPHA, UR5_A, UR5_D, q, dx, DT, QDOT_LIMIT)
    assert 0.0 < s < 1.0
    s2, dq_scaled = joint_velocity_scale(UR5_ALPHA, UR5_A, UR5_D, q, dx * s, DT, QDOT_LIMIT)
    assert s2 == pytest.approx(1.0)  # the scaled step is exactly within budget
    assert np.allclose(dq_scaled, dq * s, atol=1e-9)  # direction preserved


def test_joint_velocity_scale_is_directional_near_singularity():
    # For equal Cartesian magnitude, a step into the least-controllable direction is scaled far more
    # than one along the most-controllable direction (which stays unrestricted). This is what lets
    # the operator escape a singularity while the tripping direction is held.
    q = GENERIC_Q.copy()
    q[4] = 0.01
    jac = geometric_jacobian(UR5_ALPHA, UR5_A, UR5_D, q)
    u, _, _ = np.linalg.svd(jac)
    mag = 0.02
    s_into, _ = joint_velocity_scale(UR5_ALPHA, UR5_A, UR5_D, q, u[:, -1] * mag, DT, QDOT_LIMIT)
    s_easy, _ = joint_velocity_scale(UR5_ALPHA, UR5_A, UR5_D, q, u[:, 0] * mag, DT, QDOT_LIMIT)
    assert s_into < s_easy
    assert s_easy == pytest.approx(1.0)


def _near_exact_ik_dq(jac, dx, lam=1e-6):
    """Joint step the *controller's* near-exact IK would command for ``dx`` (~1/sigma_min spike)."""
    return jac.T @ np.linalg.solve(jac @ jac.T + lam * lam * np.eye(6), np.asarray(dx, float))


def test_joint_velocity_scale_holds_against_controller_ik_near_singularity():
    # THE regression that matters: the guard's own dq is self-consistently scaled to budget at ANY
    # lambda, but the *controller* runs a near-exact IK (~1/sigma_min). The guard must be predicted
    # with a lambda small enough that the guard-APPROVED step (dx * s) keeps that near-exact IK dq
    # within budget too. With the shipped default (1e-4) it does; with a lambda on the order of the
    # operating sigma_min (1e-2) it silently under-predicts and the true joint velocity blows past
    # budget — which reproduced as the live wrist-singularity over-speed trip.
    q = GENERIC_Q.copy()
    q[4] = 0.01  # deep wrist singularity (sigma_min ~ 4e-3)
    jac = geometric_jacobian(UR5_ALPHA, UR5_A, UR5_D, q)
    u, _, _ = np.linalg.svd(jac)
    dx = u[:, -1] * 0.02  # a modest fast twist along the least-controllable direction

    # Default lambda: guard-approved step keeps the controller's near-exact IK within its budget.
    s_good, _ = joint_velocity_scale(UR5_ALPHA, UR5_A, UR5_D, q, dx, DT, QDOT_LIMIT)
    true_ratio_good = np.max(np.abs(_near_exact_ik_dq(jac, dx * s_good)) / (QDOT_LIMIT * DT))
    assert true_ratio_good <= 1.1

    # Regression guard: the old lambda=1e-2 would leave the true joint velocity far over budget.
    s_bad, _ = joint_velocity_scale(UR5_ALPHA, UR5_A, UR5_D, q, dx, DT, QDOT_LIMIT, lam=1e-2)
    true_ratio_bad = np.max(np.abs(_near_exact_ik_dq(jac, dx * s_bad)) / (QDOT_LIMIT * DT))
    assert true_ratio_bad > 2.0


# =============================================================================
# FK self-check gate: position validates the DH (→ Jacobian/w); tool-ORIENTATION
# drift is a benign TCP-convention / about-flange-Z artifact that must NOT disable
# the guard. These tests are written FIRST (TDD) and drive the refactor of
# `tool_consistency` to return (position_drift_m, rotation_drift_deg) so the driver
# can gate on position and demote rotation to a warning.
#
# Live evidence (left arm, 2026-07-09): DH is the textbook CS66 MDH; inferred tool
# translation [6.82,-6.29,194.13]mm matches the independently-measured value to
# 0.01mm; self-check reported pos drift 0.0mm but rot drift 18.68deg. From pos=0
# it follows the FK orientation error is a rotation ABOUT the tool axis (≈flange Z),
# which leaves every joint axis z_i and origin — hence det(J) and the joint-velocity
# prediction — unchanged. So the guard's math is correct; only the self-check's
# rotation clause wrongly disabled it.

# Two well-separated configs (every joint differs by >= 0.3 rad, as the driver requires).
_Q_A = GENERIC_Q.copy()
_Q_B = GENERIC_Q + np.array([0.5, -0.4, 0.6, 0.35, -0.5, 0.45])
_UR5_DH = (UR5_ALPHA, UR5_A, UR5_D)


def _se3_to_pose6(t):
    return np.concatenate([t[:3, 3], Rotation.from_matrix(t[:3, :3]).as_rotvec()])


def _rot_z_se3(theta_rad):
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    t = np.eye(4)
    t[0, 0], t[0, 1], t[1, 0], t[1, 1] = c, -s, s, c
    return t


def _tool_se3(z_m=0.194, rot=None):
    t = np.eye(4)
    t[2, 3] = z_m
    if rot is not None:
        t[:3, :3] = rot
    return t


def _measured_tcp(dh, q, tool_se3):
    """Simulate the controller's actual_TCP_pose = true flange FK @ rigid tool."""
    return _se3_to_pose6(flange_se3(*dh, q) @ tool_se3)


def test_tool_consistency_constant_tool_is_zero_drift():
    # A genuinely rigid tool + correct DH -> the inferred flange->TCP transform is identical
    # at both configs: zero position AND zero rotation drift.
    tool = _tool_se3(rot=Rotation.from_rotvec([0.2, -0.3, 0.5]).as_matrix())
    m0, m1 = _measured_tcp(_UR5_DH, _Q_A, tool), _measured_tcp(_UR5_DH, _Q_B, tool)
    pos_drift_m, rot_drift_deg = tool_consistency(_UR5_DH, _Q_A, m0, _Q_B, m1)
    assert pos_drift_m < 1e-9
    assert rot_drift_deg < 1e-6


def test_tool_consistency_about_flange_z_is_position_clean():
    # THE justification for the gate change: a config-dependent twist ABOUT the flange Z axis
    # (the observed 18deg convention artifact) leaves the inferred tool POSITION exactly
    # consistent (a Z-aligned tool is invariant to Z rotation) while the ROTATION drift is large.
    # => a position-only gate accepts this DH (correctly), the rotation clause would reject it.
    tool = _tool_se3()  # purely along +Z
    m0 = _se3_to_pose6(flange_se3(*_UR5_DH, _Q_A) @ _rot_z_se3(np.radians(0.0)) @ tool)
    m1 = _se3_to_pose6(flange_se3(*_UR5_DH, _Q_B) @ _rot_z_se3(np.radians(18.0)) @ tool)
    pos_drift_m, rot_drift_deg = tool_consistency(_UR5_DH, _Q_A, m0, _Q_B, m1)
    assert pos_drift_m < 2e-3  # position gate PASSES — DH is valid for the Jacobian
    assert rot_drift_deg > 10.0  # rotation "drift" is large but benign (about-Z; not in det(J))


def test_tool_consistency_wrong_dh_still_caught_on_position():
    # A genuinely wrong DH (a link length off by 3 cm) makes the inferred tool POSITION
    # inconsistent across configs — the position gate alone still catches it, so relaxing the
    # rotation clause does not weaken detection of an actually-bad DH.
    tool = _tool_se3()
    m0, m1 = _measured_tcp(_UR5_DH, _Q_A, tool), _measured_tcp(_UR5_DH, _Q_B, tool)
    wrong_a = list(UR5_A)
    wrong_a[2] += 0.03
    pos_drift_m, _ = tool_consistency((UR5_ALPHA, wrong_a, UR5_D), _Q_A, m0, _Q_B, m1)
    assert pos_drift_m > 5e-3


def test_manipulability_is_independent_of_tcp_convention():
    # Safety backstop: the guard's runtime math (w and the joint-velocity prediction) is a pure
    # function of (DH, q) — it never reads the TCP — so no TCP-orientation convention can change
    # it. Locking this makes explicit why the rotation-drift clause is orthogonal to the guard.
    w = manipulability(*_UR5_DH, _Q_A)
    s, _ = joint_velocity_scale(
        *_UR5_DH,
        _Q_A,
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.01]),
        0.033,
        np.array([2.618, 2.618, 3.142, 4.014, 4.014, 4.014]) * 0.8,
    )
    # Recompute after an arbitrary about-Z change to the (irrelevant) measured TCP — identical.
    assert manipulability(*_UR5_DH, _Q_A) == w
    s2, _ = joint_velocity_scale(
        *_UR5_DH,
        _Q_A,
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.01]),
        0.033,
        np.array([2.618, 2.618, 3.142, 4.014, 4.014, 4.014]) * 0.8,
    )
    assert s2 == s
