#!/usr/bin/env python3
"""
analyze_localization_bag.py

Evaluate a recorded localization test drive (see README "Recording a test drive").

Inputs:
  --bag       the raw drive (/gps_raw, /imu/data, /imu_attitude, ...)
  --replay    NAME=BAG, optional, repeatable: output of the localization stack replayed on the drive
              (/odometry/local, /odometry/global, /odometry/gps, /tf), e.g.
                ros2 launch jimny_localization localization.launch.py use_sim_time:=true
                ros2 bag record -o replay_default --use-sim-time /odometry/local /odometry/global \
                    /odometry/gps /gps/filtered /tf /speed/twist
                ros2 bag play <drive> --clock
  --graph     road graph GeoJSON from jimny_map (default: src/jimny_map/maps/test_area.geojson)

What it measures (README "Possible next steps" 1 and "Known issues" 1-4):
  - topic rates and gaps, GPS fix quality
  - stops: gyro bias, Cube heading drift, GPS scatter vs. reported accuracy
  - loop closure and drift of PlanarOdometry and ekf_local against GPS
  - GPS cross-track distance to the OSM road edges
  - Cube heading vs. GPS course over ground (compass check), gyro scale factor
  - map -> odom jumps of ekf_global, ekf_global vs. GPS
  - junctions (OSM vertices of degree >= 3) passed, with id and time
  - suspected reversing

Needs `source install/setup.bash` (gps_msgs, cuberos message types). Writes summary.json and PNGs
to --out.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

WS_SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WS_SRC / 'centerline_follower'))
from centerline_follower.odometry import PlanarOdometry  # noqa: E402

EARTH_RADIUS_M = 6371008.8
STOP_SPEED = 0.1          # m/s, same as speed_twist min_speed
MIN_STOP_SECONDS = 3.0
HEADING_MIN_SPEED = 2.0   # m/s, course over ground reliable above this
JUNCTION_RADIUS = 10.0    # m

# reference palette (dataviz skill), light mode
C_GPS, C_PLANAR, C_LOCAL, C_GLOBAL, C_GLOBAL2 = '#2a78d6', '#eb6834', '#1baf7a', '#6250d6', '#e87ba4'
C_MAP, C_INK, C_INK2, C_GRID = '#c3c2b7', '#0b0b0b', '#52514e', '#e8e7e2'


# --------------------------------------------------------------------------------------- reading
def read_bag(uri, topics):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(uri), storage_id='sqlite3'),
                rosbag2_py.ConverterOptions('', ''))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    classes = {name: get_message(types[name]) for name in topics if name in types}
    out = {name: [] for name in topics}
    while reader.has_next():
        topic, payload, stamp_ns = reader.read_next()
        if topic in classes:
            out[topic].append((stamp_ns * 1e-9, deserialize_message(payload, classes[topic])))
    return out


def stamp(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def quat_yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


class Enu:
    """Equirectangular lat/lon -> metres east/north (fine for a few km)."""

    def __init__(self, lat0, lon0):
        self.lat0, self.lon0 = lat0, lon0
        self.kx = math.radians(1) * EARTH_RADIUS_M * math.cos(math.radians(lat0))
        self.ky = math.radians(1) * EARTH_RADIUS_M

    def __call__(self, lat, lon):
        return (np.asarray(lon) - self.lon0) * self.kx, (np.asarray(lat) - self.lat0) * self.ky


# -------------------------------------------------------------------------------------- helpers
def segments(mask, t, min_duration):
    """[(i0, i1)] index ranges where mask is true for at least min_duration."""
    out, start = [], None
    for i, m in enumerate(np.append(mask, False)):
        if m and start is None:
            start = i
        elif not m and start is not None:
            if t[i - 1] - t[start] >= min_duration:
                out.append((start, i - 1))
            start = None
    return out


def rigid_fit(src, dst):
    """R, t minimising |R src + t - dst| (2D, no scale)."""
    cs, cd = src.mean(0), dst.mean(0)
    h = (src - cs).T @ (dst - cd)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1, d]) @ u.T
    return r, cd - r @ cs


def interp_xy(t, tt, xy):
    return np.column_stack([np.interp(t, tt, xy[:, 0]), np.interp(t, tt, xy[:, 1])])


def point_segment_signed(p, a, b):
    """distance from p to segment ab and the segment direction."""
    ab = b - a
    L2 = float(ab @ ab)
    s = 0.0 if L2 == 0 else float(np.clip((p - a) @ ab / L2, 0, 1))
    q = a + s * ab
    return float(np.hypot(*(p - q))), ab, q


def stats(values):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    return {'n': int(v.size), 'mean': float(v.mean()), 'std': float(v.std()),
            'median': float(np.median(v)), 'p95': float(np.percentile(v, 95)),
            'max': float(v.max()), 'min': float(v.min())}


def r(x, nd=3):
    if isinstance(x, dict):
        return {k: r(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [r(v, nd) for v in x]
    if isinstance(x, (float, np.floating)):
        return round(float(x), nd)
    if isinstance(x, np.integer):
        return int(x)
    return x


# --------------------------------------------------------------------------------------- analysis
def load_drive(bag):
    raw = read_bag(bag, ['/gps_raw', '/gps/fix', '/imu/data', '/imu_attitude', '/heartbeat',
                         '/rc_inputs'])
    # GPS: one row per new fix (cuberos may repeat samples)
    rows, last = [], None
    for tb, m in raw['/gps_raw']:
        key = (m.latitude, m.longitude, m.speed, m.track)
        if key == last:
            continue
        last = key
        rows.append((stamp(m), tb, m.latitude, m.longitude, m.speed, m.track, m.err_horz,
                     m.err_speed, m.status.status, m.status.satellites_visible, m.hdop))
    gps = np.array(rows)
    imu = np.array([(stamp(m), tb, quat_yaw(m.orientation), m.angular_velocity.z)
                    for tb, m in raw['/imu/data']])
    imu = imu[np.argsort(imu[:, 0])]
    _, keep = np.unique(imu[:, 0], return_index=True)
    imu = imu[keep]
    att = np.array([(stamp(m), m.angular_velocity.z) for _, m in raw['/imu_attitude']])
    hb = [(tb, m.mode, m.armed) for tb, m in raw['/heartbeat']]
    timing = {}
    for topic, msgs in raw.items():
        if len(msgs) < 2:
            continue
        tb = np.array([x[0] for x in msgs])
        dt = np.diff(tb)
        timing[topic] = {'count': len(msgs), 'rate_hz': (len(msgs) - 1) / (tb[-1] - tb[0]),
                         'median_dt': float(np.median(dt)), 'max_gap': float(dt.max()),
                         'gaps_over_0.5s': int((dt > 0.5).sum())}
    gdt = np.diff(gps[:, 0])
    timing['/gps_raw (unique fixes)'] = {'count': len(gps), 'rate_hz': (len(gps) - 1) / (gps[-1, 0] - gps[0, 0]),
                                         'median_dt': float(np.median(gdt)), 'max_gap': float(gdt.max()),
                                         'gaps_over_0.5s': int((gdt > 0.5).sum())}
    return gps, imu, att, hb, timing


def run_planar(gps_t, speed, imu_t, wz, sign):
    odo = PlanarOdometry(yaw_rate_sign=sign)
    events = [(t, 0, v) for t, v in zip(gps_t, speed)] + [(t, 1, w) for t, w in zip(imu_t, wz)]
    events.sort(key=lambda e: (e[0], e[1]))
    out, restarts, last_epoch = [], 0, 0
    for t, kind, v in events:
        if kind == 0:
            odo.update_speed(v, t)
            continue
        odo.update_yaw_rate(v, t)
        if odo.epoch != last_epoch:
            restarts += odo.epoch - last_epoch
            last_epoch = odo.epoch
        pose, _ = odo.pose(t)
        if pose is not None:
            out.append((t, *pose))
    return np.array(out), restarts


def load_replay(uri):
    raw = read_bag(uri, ['/odometry/local', '/odometry/global', '/odometry/gps', '/tf'])

    def odom(topic):
        a = np.array([(stamp(m), m.pose.pose.position.x, m.pose.pose.position.y,
                       quat_yaw(m.pose.pose.orientation)) for _, m in raw[topic]])
        return a[np.argsort(a[:, 0])] if len(a) else a
    map_odom = []
    for _, m in raw['/tf']:
        for tr in m.transforms:
            if tr.header.frame_id == 'map' and tr.child_frame_id == 'odom':
                map_odom.append((stamp(tr), tr.transform.translation.x, tr.transform.translation.y,
                                 quat_yaw(tr.transform.rotation)))
    mo = np.array(map_odom)
    return {'local': odom('/odometry/local'), 'global': odom('/odometry/global'),
            'gps': odom('/odometry/gps'), 'map_odom': mo[np.argsort(mo[:, 0])] if len(mo) else mo}


def load_graph(path, enu):
    gj = json.loads(Path(path).read_text())
    edges, nodes = [], []
    for f in gj['features']:
        p = f['properties']
        if f['geometry']['type'] == 'LineString':
            c = np.array(f['geometry']['coordinates'])
            x, y = enu(c[:, 1], c[:, 0])
            edges.append((p['id'], np.column_stack([x, y]), p))
        elif p.get('kind') == 'node':
            lon, lat = f['geometry']['coordinates'][:2]
            x, y = enu(lat, lon)
            nodes.append((p['id'], float(x), float(y), p.get('degree', 0)))
    return edges, nodes


def nearest_edge(p, edges):
    best = (1e9, None, None, None)
    for eid, xy, _ in edges:
        lo, hi = xy.min(0) - best[0], xy.max(0) + best[0]
        if (p < lo).any() or (p > hi).any():
            continue
        for a, b in zip(xy[:-1], xy[1:]):
            d, ab, q = point_segment_signed(p, a, b)
            if d < best[0]:
                best = (d, eid, ab, q)
    return best


def analyze(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {'bag': str(args.bag)}

    gps, imu, att, hb, timing = load_drive(args.bag)
    summary['timing'] = timing
    gt, glat, glon, gspd, gtrack = gps[:, 0], gps[:, 2], gps[:, 3], gps[:, 4], gps[:, 5]
    gerr, gstatus, gsats, ghdop = gps[:, 6], gps[:, 8], gps[:, 9], gps[:, 10]
    t0 = gt[0]
    enu = Enu(glat[0], glon[0])
    gx, gy = enu(glat, glon)
    gxy = np.column_stack([gx, gy])
    moving = gspd >= STOP_SPEED
    dist = np.concatenate([[0], np.cumsum(np.hypot(np.diff(gx), np.diff(gy)) * moving[1:])])
    course = wrap(np.radians(90.0 - gtrack))       # GPS track (deg cw from north) -> ENU yaw
    summary['drive'] = {
        'duration_s': float(gt[-1] - t0), 'distance_m': float(dist[-1]),
        'max_speed_mps': float(gspd.max()), 'mean_moving_speed_mps': float(gspd[moving].mean()),
        'gps_status_values': {str(int(k)): int(v) for k, v in zip(*np.unique(gstatus, return_counts=True))},
        'satellites_visible': stats(gsats), 'hdop': stats(ghdop), 'reported_h_acc_m': stats(gerr),
        'modes': sorted({m for _, m, _ in hb}),
    }

    # GPS antenna -> base_link (rear axle): antenna is gps_x ahead along the heading
    held = course.copy()
    for i in range(len(held)):
        if gspd[i] < 0.5 and i > 0:
            held[i] = held[i - 1]
    bxy = gxy - args.gps_x * np.column_stack([np.cos(held), np.sin(held)])

    it, iyaw, iwz = imu[:, 0], imu[:, 2], imu[:, 3]

    # ------------------------------------------------------------------- stops
    stops = []
    for i0, i1 in segments(~moving, gt, MIN_STOP_SECONDS):
        ta, tb = gt[i0], gt[i1]
        sel = (it >= ta) & (it <= tb)
        if sel.sum() < 5:
            continue
        p = gxy[i0:i1 + 1]
        dyaw_cube = float(np.degrees(wrap(iyaw[sel][-1] - iyaw[sel][0])))
        stops.append({
            't_start': ta - t0, 'duration_s': tb - ta,
            'gyro_wz_mean_dps': float(np.degrees(iwz[sel].mean())),
            'gyro_wz_std_dps': float(np.degrees(iwz[sel].std())),
            'gyro_yaw_integrated_deg': float(np.degrees(np.trapz(iwz[sel], it[sel]))),
            'cube_yaw_change_deg': dyaw_cube,
            'gps_scatter_std_m': float(np.hypot(*p.std(0))),
            'gps_wander_max_m': float(np.hypot(*(p - p.mean(0)).T).max()),
            'reported_h_acc_m': float(gerr[i0:i1 + 1].mean()),
        })
    bias = [s['gyro_wz_mean_dps'] for s in stops]
    w = [s['duration_s'] for s in stops]
    summary['stops'] = {'count': len(stops), 'list': stops,
                        'gyro_bias_dps_weighted': float(np.average(bias, weights=w)) if stops else None,
                        'total_stop_s': float(sum(w))}

    # ------------------------------------------------------------------- yaw rate sanity / scale
    aw = np.interp(it, att[:, 0], att[:, 1])
    summary['imu_sign_check'] = {
        'corr_imu_data_vs_imu_attitude_wz': float(np.corrcoef(iwz, aw)[0, 1]),
        'expected': -1.0}
    # gyro heading vs GPS course over 2 s windows while driving fast
    gyro_heading = np.concatenate([[0], np.cumsum(0.5 * (iwz[1:] + iwz[:-1]) * np.diff(it))])
    gh_at_gps = np.interp(gt, it, gyro_heading)
    win = []
    for i in range(len(gt)):
        j = np.searchsorted(gt, gt[i] + 2.0)
        if j >= len(gt) or gspd[i:j + 1].min() < HEADING_MIN_SPEED:
            continue
        dc = float(wrap(course[j] - course[i]))
        dg = gh_at_gps[j] - gh_at_gps[i]
        if abs(dc) > math.radians(10):
            win.append((dc, dg))
    if len(win) > 10:
        w_arr = np.array(win)
        scale = float((w_arr[:, 0] @ w_arr[:, 1]) / (w_arr[:, 1] @ w_arr[:, 1]))
        summary['gyro_scale'] = {'gps_course_change_per_gyro_change': scale, 'windows': len(win)}
    else:
        summary['gyro_scale'] = None

    # ------------------------------------------------------------------- heading: Cube vs course
    cube_at_gps = np.interp(gt, it, np.unwrap(iyaw))
    wz_at_gps = np.interp(gt, it, iwz)
    hsel = (gspd >= HEADING_MIN_SPEED) & (np.abs(wz_at_gps) < math.radians(5))
    hdiff = wrap(cube_at_gps - course)
    hd = hdiff[hsel]
    hc = course[hsel]
    fit = None
    if hd.size > 20:
        A = np.column_stack([np.ones_like(hc), np.cos(hc), np.sin(hc)])
        coef, *_ = np.linalg.lstsq(A, hd, rcond=None)
        resid = hd - A @ coef
        fit = {'offset_deg': float(np.degrees(coef[0])),
               'sinusoid_amplitude_deg': float(np.degrees(np.hypot(coef[1], coef[2]))),
               'residual_std_deg': float(np.degrees(resid.std()))}
    cube_fast = np.degrees(wrap(cube_at_gps[gspd >= HEADING_MIN_SPEED]))
    # gyro heading drift: GPS course minus integrated /imu/data yaw rate (slope = bias, steps = events)
    dsel = gspd >= HEADING_MIN_SPEED
    gyro_drift = np.degrees(np.unwrap(wrap(course[dsel] - gh_at_gps[dsel])))
    gyro_drift -= gyro_drift[0]
    drift_t = gt[dsel] - t0
    summary['heading'] = {'cube_minus_course_deg': stats(np.degrees(hd)), 'fit': fit,
                          'samples': int(hsel.sum()),
                          'cube_yaw_range_while_driving_deg': [float(np.percentile(cube_fast, 5)),
                                                               float(np.percentile(cube_fast, 95))],
                          'gyro_heading_drift_end_deg': float(gyro_drift[-1]) if gyro_drift.size else None,
                          'gyro_heading_drift_max_deg': float(np.abs(gyro_drift).max()) if gyro_drift.size else None}

    # ------------------------------------------------------------------- reversing
    off = wrap(course - gh_at_gps)
    ref = np.median(off[gspd >= 1.0]) if (gspd >= 1.0).any() else 0
    # drift-tolerant: compare against rolling median over 60 s
    rev = []
    for i in np.where(gspd >= 0.5)[0]:
        sel = (np.abs(gt - gt[i]) < 30) & (gspd >= 1.0)
        base = float(np.median(np.unwrap(off[sel]))) if sel.sum() > 5 else ref
        rev.append(abs(float(wrap(off[i] - base))) > math.radians(120))
    rev_idx = np.where(gspd >= 0.5)[0][np.array(rev, dtype=bool)] if rev else np.array([], int)
    rev_mask = np.zeros(len(gt), bool)
    rev_mask[rev_idx] = True
    summary['reversing'] = [{'t_start': gt[a] - t0, 'duration_s': gt[b] - gt[a]}
                            for a, b in segments(rev_mask, gt, 1.0)]

    # ------------------------------------------------------------------- PlanarOdometry
    planar, planar_restarts = run_planar(gt, gspd, it, iwz, +1.0)
    planar_att, _ = run_planar(gt, gspd, att[:, 0], att[:, 1], -1.0)
    tracks = {'PlanarOdometry': planar}
    replays = {}
    for spec in args.replay or []:
        name, uri = spec.split('=', 1)
        replays[name] = load_replay(uri)
    first_replay = next(iter(replays.values()), None)
    if first_replay is not None and len(first_replay['local']):
        tracks['ekf_local'] = first_replay['local']

    # loop closure: mean position over first / last stop (or first / last fix)
    def window(ta, tb, tt):
        return (tt >= ta) & (tt <= tb)
    first_stop = stops[0] if stops and stops[0]['t_start'] < 5 else None
    last_stop = stops[-1] if stops and stops[-1]['t_start'] + stops[-1]['duration_s'] > gt[-1] - t0 - 5 else None
    ta0, ta1 = (t0, t0 + first_stop['duration_s']) if first_stop else (t0, t0 + 1)
    tb0, tb1 = (t0 + last_stop['t_start'], gt[-1]) if last_stop else (gt[-1] - 1, gt[-1])
    loop = {'start_window_s': [ta0 - t0, ta1 - t0], 'end_window_s': [tb0 - t0, tb1 - t0]}
    gs, ge = bxy[window(ta0, ta1, gt)].mean(0), bxy[window(tb0, tb1, gt)].mean(0)
    loop['gps'] = {'start_to_end_m': float(np.hypot(*(ge - gs)))}
    for name, tr in tracks.items():
        s = tr[window(ta0, ta1, tr[:, 0]), 1:3]
        e = tr[window(tb0, tb1, tr[:, 0]), 1:3]
        if not len(s):
            s = tr[:1, 1:3]
        if not len(e):
            e = tr[-1:, 1:3]
        d = float(np.hypot(*(e.mean(0) - s.mean(0))))
        loop[name] = {'start_to_end_m': d, 'percent_of_distance': 100 * d / dist[-1]}
    summary['loop_closure'] = loop

    # ------------------------------------------------------------------- odom vs GPS (aligned)
    drift = {}
    aligned = {}
    for name, tr in tracks.items():
        tt = tr[:, 0]
        sel = (gt >= tt[0]) & (gt <= tt[-1])
        ref_xy, t_ref, d_ref = bxy[sel], gt[sel], dist[sel]
        oxy = interp_xy(t_ref, tt, tr[:, 1:3])
        fitsel = (d_ref - d_ref[0]) <= args.align_distance
        fitsel &= d_ref > d_ref[0] + 2.0
        if fitsel.sum() < 5:
            fitsel = np.ones_like(fitsel)
        R, tvec = rigid_fit(oxy[fitsel], ref_xy[fitsel])
        oxy_al = oxy @ R.T + tvec
        full_al = tr[:, 1:3] @ R.T + tvec
        err = np.hypot(*(oxy_al - ref_xy).T)
        aligned[name] = (full_al, t_ref, d_ref, err)
        marks = {}
        for D in (100, 250, 500, 1000, 2000):
            k = np.searchsorted(d_ref, D)
            if k < len(err):
                marks[f'{D}m'] = float(err[k])
        drift[name] = {'error_m': stats(err), 'error_at_distance': marks,
                       'error_at_end_m': float(err[-1]), 'align_rotation_deg': float(np.degrees(math.atan2(R[1, 0], R[0, 0])))}
    drift['PlanarOdometry']['restarts'] = planar_restarts
    pa = interp_xy(planar[:, 0], planar_att[:, 0], planar_att[:, 1:3])
    drift['PlanarOdometry_imu_attitude_vs_imu_data_max_m'] = float(np.hypot(*(pa - planar[:, 1:3]).T).max())
    if 'ekf_local' in tracks:
        el = interp_xy(planar[:, 0], tracks['ekf_local'][:, 0], tracks['ekf_local'][:, 1:3])
        drift['ekf_local_vs_planar_max_m'] = float(np.hypot(*(el - planar[:, 1:3]).T).max())
        # yaw accumulated by ekf_local while stopped (known issue 4)
        L = tracks['ekf_local']
        yl = np.unwrap(L[:, 3])
        stop_yaw = []
        for s in stops:
            a, b = t0 + s['t_start'], t0 + s['t_start'] + s['duration_s']
            sel = window(a, b, L[:, 0])
            if sel.sum() > 2:
                stop_yaw.append(float(np.degrees(yl[sel][-1] - yl[sel][0])))
            else:
                stop_yaw.append(None)
            s['ekf_local_yaw_change_deg'] = stop_yaw[-1]
        drift['ekf_local_yaw_while_stopped_total_deg'] = float(np.nansum([v for v in stop_yaw if v is not None]))
    summary['odometry_vs_gps'] = drift

    # ------------------------------------------------------------------- replays: global
    glob = {}
    for name, rp in replays.items():
        g = rp['global']
        if not len(g):
            continue
        # ekf_global and /odometry/gps live in the navsat map frame; compare via /odometry/gps
        og = rp['gps']
        res = {}
        if len(og):
            gi = interp_xy(og[:, 0], g[:, 0], g[:, 1:3])
            e = np.hypot(*(gi - og[:, 1:3]).T)
            res['ekf_global_vs_odometry_gps_m'] = stats(e)
        mo = rp['map_odom']
        if len(mo) > 1:
            dxy = np.hypot(np.diff(mo[:, 1]), np.diff(mo[:, 2]))
            dyaw = np.degrees(np.abs(wrap(np.diff(mo[:, 3]))))
            dt = np.diff(mo[:, 0])
            res['map_odom_step_m'] = stats(dxy)
            res['map_odom_jumps_over_0.3m'] = int((dxy > 0.3).sum())
            res['map_odom_step_yaw_deg_max'] = float(dyaw.max())
            res['map_odom_total_yaw_change_deg'] = float(np.degrees(wrap(mo[-1, 3] - mo[0, 3])))
            res['map_odom_largest'] = [{'t': float(mo[k + 1, 0] - t0), 'step_m': float(dxy[k]),
                                        'dt_s': float(dt[k])} for k in np.argsort(dxy)[-5:][::-1]]
        # heading of ekf_global vs GPS course
        gy_at = np.interp(gt, g[:, 0], np.unwrap(g[:, 3]))
        hdg = np.degrees(wrap(gy_at - course))[hsel & (gt >= g[0, 0]) & (gt <= g[-1, 0])]
        res['ekf_global_yaw_minus_course_deg'] = stats(hdg)
        glob[name] = res
    summary['ekf_global'] = glob

    # ------------------------------------------------------------------- OSM
    edges, nodes = load_graph(args.graph, enu)
    xt = []
    sub = np.where(gspd >= 1.0)[0][::2]
    for i in sub:
        d, eid, ab, q = nearest_edge(gxy[i], edges)
        if eid is None:
            continue
        side = math.copysign(1.0, ab[0] * (gxy[i, 1] - q[1]) - ab[1] * (gxy[i, 0] - q[0]))
        travel_dir = np.array([math.cos(course[i]), math.sin(course[i])])
        along = 1.0 if travel_dir @ ab >= 0 else -1.0
        # positive = GPS left of the road centre as seen in the direction of travel
        xt.append((gt[i], d * side * along, d, eid, along))
    xt = np.array(xt)
    if len(xt):
        on_map = xt[:, 2] <= args.off_map
        signed = xt[on_map, 1]
        per_edge = {}
        for eid in np.unique(xt[on_map, 3]):
            for along in (1.0, -1.0):
                m = on_map & (xt[:, 3] == eid) & (xt[:, 4] == along)
                if m.sum() >= 10:
                    per_edge[f'{int(eid)}{"+" if along > 0 else "-"}'] = {
                        'n': int(m.sum()), 'mean_left_m': float(xt[m, 1].mean()), 'std_m': float(xt[m, 1].std())}
        summary['gps_vs_osm'] = {
            'signed_left_of_centre_m': stats(signed), 'abs_m': stats(np.abs(signed)),
            'off_map_fraction': float(1 - on_map.mean()), 'off_map_threshold_m': args.off_map,
            'per_edge_direction': per_edge}
    # junctions passed
    passes = []
    for nid, nx, ny, deg in nodes:
        if deg < 3:
            continue
        d = np.hypot(gx - nx, gy - ny)
        near = d <= JUNCTION_RADIUS
        for a, b in segments(near, gt, 0.0):
            k = a + int(np.argmin(d[a:b + 1]))
            passes.append({'node_id': int(nid), 'degree': int(deg), 't': float(gt[k] - t0),
                           'min_dist_m': float(d[k]), 'distance_driven_m': float(dist[k])})
    passes.sort(key=lambda p: p['t'])
    summary['junctions_passed'] = passes

    (out / 'summary.json').write_text(json.dumps(r(summary), indent=2))

    if not args.no_plots:
        plots(out, summary, gt, t0, gxy, bxy, gspd, dist, edges, nodes, passes, aligned, replays,
              hc, hd, fit, it, iwz, stops, xt, args, (drift_t, gyro_drift))
    return summary


# ------------------------------------------------------------------------------------------ plots
def style(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor('#fcfcfb')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(C_INK2)
    ax.tick_params(colors=C_INK2, labelsize=9)
    ax.grid(True, color=C_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, loc='left', color=C_INK, fontsize=11)
    if xlabel:
        ax.set_xlabel(xlabel, color=C_INK2, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=C_INK2, fontsize=9)


def plots(out, summary, gt, t0, gxy, bxy, gspd, dist, edges, nodes, passes, aligned, replays,
          hc, hd, fit, it, iwz, stops, xt, args, gyro_drift):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'figure.facecolor': '#fcfcfb', 'savefig.facecolor': '#fcfcfb'})

    # 1. map
    fig, ax = plt.subplots(figsize=(10, 10))
    pad = 60
    lo, hi = gxy.min(0) - pad, gxy.max(0) + pad
    for _, xy, _p in edges:
        ax.plot(xy[:, 0], xy[:, 1], color=C_MAP, linewidth=6, solid_capstyle='round', zorder=1)
    ax.plot(bxy[:, 0], bxy[:, 1], color=C_GPS, linewidth=2, label='GPS (rear axle)', zorder=4)
    colors = {'PlanarOdometry': C_PLANAR, 'ekf_local': C_LOCAL}
    for name, (al, *_rest) in aligned.items():
        ax.plot(al[:, 0], al[:, 1], color=colors.get(name, C_INK2), linewidth=2,
                label=f'{name} (aligned on first {args.align_distance:.0f} m)', zorder=3)
    shown = set()
    for p in passes:
        n = next(n for n in nodes if n[0] == p['node_id'])
        if n[0] in shown:
            continue
        shown.add(n[0])
        ax.plot(n[1], n[2], 'o', ms=8, color='#fcfcfb', mec=C_INK, mew=1.5, zorder=5)
        ax.annotate(str(n[0]), (n[1], n[2]), xytext=(6, 6), textcoords='offset points',
                    fontsize=9, color=C_INK, zorder=6)
    ax.plot(*bxy[0], 's', ms=9, color=C_GPS, mec='#fcfcfb', zorder=6)
    ax.annotate('start', bxy[0], xytext=(8, -12), textcoords='offset points', color=C_INK2)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect('equal')
    style(ax, 'Drive on the OSM road graph (circles = junctions passed, with id)', 'east [m]', 'north [m]')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.8), frameon=True, facecolor='#fcfcfb', edgecolor=C_GRID)
    fig.tight_layout()
    fig.savefig(out / 'map.png', dpi=110)
    plt.close(fig)

    # 1b. global replays on the map
    if replays:
        fig, ax = plt.subplots(figsize=(10, 10))
        for _, xy, _p in edges:
            ax.plot(xy[:, 0], xy[:, 1], color=C_MAP, linewidth=6, solid_capstyle='round', zorder=1)
        ax.plot(bxy[:, 0], bxy[:, 1], color=C_GPS, linewidth=2, label='GPS (rear axle)', zorder=4)
        cols = [C_GLOBAL, C_PLANAR]
        for k, (name, rp) in enumerate(replays.items()):
            g, og = rp['global'], rp['gps']
            if not len(g) or not len(og):
                continue
            # navsat map frame -> our ENU: fit /odometry/gps onto the GPS track (pure translation+rot)
            gpos = interp_xy(og[:, 0], gt, gxy)
            R, tv = rigid_fit(og[:, 1:3], gpos)
            gg = g[:, 1:3] @ R.T + tv
            ax.plot(gg[:, 0], gg[:, 1], color=cols[k % 2], linewidth=1.6,
                    label=f'ekf_global [{name}] (map frame rotated {np.degrees(math.atan2(R[1, 0], R[0, 0])):.0f}° to fit)',
                    zorder=3)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_aspect('equal')
        style(ax, 'ekf_global replays vs GPS', 'east [m]', 'north [m]')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.8), frameon=True, facecolor='#fcfcfb', edgecolor=C_GRID)
        fig.tight_layout()
        fig.savefig(out / 'map_global.png', dpi=110)
        plt.close(fig)

    # 2. odom error vs distance
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for name, (_al, _t, d, err) in aligned.items():
        ax.plot(d, err, color=colors.get(name, C_INK2), linewidth=2, label=name)
        ax.annotate(f'{err[-1]:.1f} m', (d[-1], err[-1]), xytext=(4, 0), textcoords='offset points',
                    color=C_INK, va='center')
    style(ax, 'Dead-reckoning error against GPS', 'distance driven [m]', 'position error [m]')
    ax.legend(loc='upper left', frameon=False)
    fig.tight_layout()
    fig.savefig(out / 'odom_error.png', dpi=110)
    plt.close(fig)

    # 3. heading: Cube compass yaw against GPS course (should lie on the diagonal)
    if hd.size:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.4), gridspec_kw={'width_ratios': [1, 1.4]})
        course_deg = np.degrees(hc) % 360
        cube_deg = (course_deg + np.degrees(hd)) % 360
        a1.plot([0, 360], [0, 360], color=C_INK2, linewidth=1, label='perfect compass')
        a1.scatter(course_deg, cube_deg, s=8, color=C_GPS, alpha=0.5, linewidths=0, label='Cube yaw')
        a1.set_xlim(0, 360)
        a1.set_ylim(0, 360)
        a1.set_xticks(range(0, 361, 90))
        a1.set_yticks(range(0, 361, 90))
        a1.set_aspect('equal')
        style(a1, 'Cube heading vs GPS course', 'GPS course, ENU [deg]', 'Cube yaw, ENU [deg]')
        a1.legend(loc='upper left', frameon=False)
        dt_, dd_ = gyro_drift
        for s_ in stops:
            a2.axvspan(s_['t_start'], s_['t_start'] + s_['duration_s'], color=C_GRID, zorder=0)
        a2.plot(dt_, dd_, '.', ms=2.5, color=C_LOCAL)
        a2.axhline(0, color=C_INK2, linewidth=1)
        style(a2, 'GPS course minus integrated yaw rate', 'time since start [s] (grey = stops)',
              'heading drift [deg]')
        fig.tight_layout()
        fig.savefig(out / 'heading.png', dpi=110)
        plt.close(fig)

    # 4. cross track
    if len(xt):
        on = xt[:, 2] <= args.off_map
        fig, ax = plt.subplots(figsize=(10, 4.2))
        ax.hist(xt[on, 1], bins=np.arange(-args.off_map, args.off_map + 0.25, 0.25), color=C_GPS,
                edgecolor='#fcfcfb', linewidth=1)
        ax.axvline(0, color=C_INK2, linewidth=1)
        m = float(np.median(xt[on, 1]))
        ax.axvline(m, color=C_INK, linewidth=2)
        ax.annotate(f'median {m:+.2f} m', (m, ax.get_ylim()[1] * 0.92), xytext=(6, 0),
                    textcoords='offset points', color=C_INK)
        style(ax, 'GPS distance from OSM road centre (positive = left of centre in driving direction)',
              'signed distance [m]', 'samples')
        fig.tight_layout()
        fig.savefig(out / 'cross_track.png', dpi=110)
        plt.close(fig)

    # 5. timeline
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    for s in stops:
        for a in (a1, a2):
            a.axvspan(s['t_start'], s['t_start'] + s['duration_s'], color=C_GRID, zorder=0)
    a1.plot(gt - t0, gspd, color=C_GPS, linewidth=1.5)
    style(a1, 'GPS speed (grey bands = stops)', None, 'speed [m/s]')
    a2.plot(it - t0, np.degrees(iwz), color=C_LOCAL, linewidth=1)
    style(a2, 'IMU yaw rate (/imu/data, left turn positive)', 'time since start [s]', 'yaw rate [deg/s]')
    fig.tight_layout()
    fig.savefig(out / 'timeline.png', dpi=110)
    plt.close(fig)

    # 6. map -> odom
    if replays:
        fig, ax = plt.subplots(figsize=(10, 4.2))
        cols = [C_GLOBAL, C_PLANAR]
        for k, (name, rp) in enumerate(replays.items()):
            mo = rp['map_odom']
            if len(mo) < 2:
                continue
            step = np.hypot(np.diff(mo[:, 1]), np.diff(mo[:, 2]))
            ax.plot(mo[1:, 0] - t0, step, color=cols[k % 2], linewidth=1.2, label=name)
        style(ax, 'map → odom change per TF update (ekf_global corrections)', 'time since start [s]', 'step [m]')
        ax.legend(loc='upper right', frameon=False)
        fig.tight_layout()
        fig.savefig(out / 'map_odom.png', dpi=110)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bag', required=True)
    ap.add_argument('--replay', action='append', help='NAME=BAG of a localization replay')
    ap.add_argument('--graph', default=str(WS_SRC / 'jimny_map' / 'maps' / 'test_area.geojson'))
    ap.add_argument('--out', default=None, help='output directory (default: <bag>/eval)')
    ap.add_argument('--gps-x', type=float, default=0.93, help='antenna ahead of the rear axle [m]')
    ap.add_argument('--align-distance', type=float, default=50.0,
                    help='odometry is aligned to GPS on the first N metres driven')
    ap.add_argument('--off-map', type=float, default=8.0,
                    help='GPS further than this from any OSM edge counts as off the map [m]')
    ap.add_argument('--no-plots', action='store_true')
    args = ap.parse_args()
    args.out = args.out or str(Path(args.bag) / 'eval')
    s = analyze(args)
    print(json.dumps(r({k: v for k, v in s.items() if k not in ('timing',)}, 2), indent=1)[:12000])


if __name__ == '__main__':
    main()
