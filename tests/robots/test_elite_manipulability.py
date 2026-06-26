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
    geometric_jacobian,
    manipulability,
)

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
