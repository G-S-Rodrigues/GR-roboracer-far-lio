# Copyright 2026 Guilherme Rodrigues
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
"""Synthetic F1TENTH oval shared by the launch tests (FARLIO_1610, FARLIO_1820..1880): walls, centerline, map .pcd,
ray-cast 64-beam scans at their declared angles, wheel speeds. Extracted from test_f1tenth_synthetic_lap.py (second
user: part 4b); behaviour unchanged. Not a test file."""
import math
import os

import numpy as np

LASER_X, LASER_Z = 0.10355, 0.031   # base_link -> laser (URDF, plan table)
STRAIGHT, RADIUS, HALF_WIDTH = 8.0, 3.0, 1.1
# NUM_BEAMS and SIGMA are read from the environment exactly as FARLIO_1610 reads them (case, FARLIO_LAP_BEAMS,
# FARLIO_LAP_NOISE_X); 64 beams and 3 cm for every other user.
NUM_BEAMS = int(os.environ.get('FARLIO_LAP_BEAMS', os.environ.get('FARLIO_LAP_CASE', 'kin64').lstrip('kinh')))
FOV, RANGE_MIN, RANGE_MAX = 4.7, 0.06, 10.0
SIGMA = 0.03 * float(os.environ.get('FARLIO_LAP_NOISE_X', '1'))
WHEEL_R, HALF_TRACK, LR, LF = 0.058, 0.118, 0.17145, 0.33 - 0.17145   # wheel radius, half track, CoG to axles
LAP_M = 2.0 * STRAIGHT + 2.0 * math.pi * RADIUS


def stadium_wall(offset, spacing):
    """Closed counter-clockwise polyline of the stadium wall at `offset` from the centerline."""
    r, half = RADIUS + offset, STRAIGHT / 2.0
    n_arc = max(int(math.pi * r / spacing), 1)
    a = np.linspace(-math.pi / 2.0, math.pi / 2.0, n_arc, endpoint=False)
    right = np.stack([half + r * np.cos(a), r * np.sin(a)], axis=1)
    a = np.linspace(math.pi / 2.0, 3.0 * math.pi / 2.0, n_arc, endpoint=False)
    left = np.stack([-half + r * np.cos(a), r * np.sin(a)], axis=1)
    xs = np.arange(-half, half, spacing)
    bottom = np.stack([xs, np.full_like(xs, -r)], axis=1)
    top = np.stack([-xs, np.full_like(xs, r)], axis=1)
    return np.concatenate([bottom, right, top, left])


def centerline_pose(s):
    """(x, y, yaw, curvature) at arc length s, counter-clockwise from (-4, -3), shifted so s = 0 is the origin."""
    half, arc = STRAIGHT / 2.0, math.pi * RADIUS
    u = s % LAP_M
    if u < STRAIGHT:
        x, y, yaw, k = -half + u, -RADIUS, 0.0, 0.0
    elif u < STRAIGHT + arc:
        a = -math.pi / 2.0 + (u - STRAIGHT) / RADIUS
        x, y, yaw, k = half + RADIUS * math.cos(a), RADIUS * math.sin(a), a + math.pi / 2.0, 1.0 / RADIUS
    elif u < 2.0 * STRAIGHT + arc:
        x, y, yaw, k = half - (u - STRAIGHT - arc), RADIUS, math.pi, 0.0
    else:
        a = math.pi / 2.0 + (u - 2.0 * STRAIGHT - arc) / RADIUS
        x, y, yaw, k = -half + RADIUS * math.cos(a), RADIUS * math.sin(a), a + math.pi / 2.0, 1.0 / RADIUS
    return x + half, y + RADIUS, yaw, k


def _segments(spacing):
    segs = []
    for off in (-HALF_WIDTH, HALF_WIDTH):
        w = stadium_wall(off, spacing) + np.array([STRAIGHT / 2.0, RADIUS])   # same shift as centerline_pose
        segs.append(np.stack([w, np.roll(w, -1, axis=0)], axis=1))
    return np.concatenate(segs)   # (M, 2, 2): segment start and end


def write_map_pcd(path, transform=None):
    """Write the oval's walls as an ascii .pcd at the laser height.

    `transform` is None (identity) or (tx, ty, yaw_rad): the map frame is the oval's frame moved by that rigid
    transform, so a car that starts at the oval's (0, 0, 0) starts at (tx, ty, yaw) in the map.
    """
    pts = np.concatenate([stadium_wall(off, 0.05) + np.array([STRAIGHT / 2.0, RADIUS])
                          for off in (-HALF_WIDTH, HALF_WIDTH)])
    if transform is not None:
        tx, ty, th = transform
        c, s = math.cos(th), math.sin(th)
        pts = pts @ np.array([[c, s], [-s, c]]) + np.array([tx, ty])
    with open(path, 'w') as f:
        f.write('# .PCD v0.7\nVERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\n')
        f.write(f'WIDTH {len(pts)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(pts)}\nDATA ascii\n')
        for x, y in pts:
            f.write(f'{x:.5f} {y:.5f} {LASER_Z:.5f} 0\n')


def cast_scan(segs, x, y, yaw, rng):
    """Ranges of 64 beams at their declared angles from the laser pose; inf when nothing is hit in range."""
    ox, oy = x + LASER_X * math.cos(yaw), y + LASER_X * math.sin(yaw)
    ang = yaw + (-FOV / 2.0 + np.arange(NUM_BEAMS) * FOV / (NUM_BEAMS - 1))
    d = np.stack([np.cos(ang), np.sin(ang)], axis=1)                    # (N, 2)
    a, e = segs[:, 0, :], segs[:, 1, :] - segs[:, 0, :]                 # (M, 2)
    den = d[:, None, 0] * e[None, :, 1] - d[:, None, 1] * e[None, :, 0]  # (N, M)
    ao = a - np.array([ox, oy])
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (ao[None, :, 0] * e[None, :, 1] - ao[None, :, 1] * e[None, :, 0]) / den
        u = (ao[None, :, 0] * d[:, None, 1] - ao[None, :, 1] * d[:, None, 0]) / den
    ok = (np.abs(den) > 1e-12) & (t > 0.0) & (u >= 0.0) & (u <= 1.0)
    r = np.where(ok, t, np.inf).min(axis=1)
    r = r + SIGMA * rng.standard_normal(NUM_BEAMS)
    r[(r > RANGE_MAX) | (r < RANGE_MIN)] = np.inf
    return r


def _wheel(px, py, v, omega):
    """Wheel speed (rad/s) of the wheel at (px, py) from the CoG: the rigid-body speed of that point."""
    if abs(omega) < 1e-9:
        return v / WHEEL_R
    return omega * math.hypot(px, py - v / omega) / WHEEL_R


def _yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
