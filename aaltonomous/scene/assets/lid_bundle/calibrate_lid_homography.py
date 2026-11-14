#!/usr/bin/env python3
"""Compute a pixel -> STL local XY (mm) homography from FOUR measured lid corners.

Usage:
  python calibrate_lid_homography.py --config lid_geometry_config.json \
      --corners camera_corners.json --output lid_homography.json

Requires only numpy. Undistort image using camera intrinsics first.
This produces a planar mapping; it does NOT produce a 3D robot/world pose.
"""
import argparse
import json
from pathlib import Path
import numpy as np

KEYS = ['minus_x_minus_y', 'plus_x_minus_y', 'plus_x_plus_y', 'minus_x_plus_y']

def normalized_dlt(source, target):
    """Numerically stable DLT: x_dst ~ H @ x_src. Both are Nx2."""
    def norm(points):
        mean = points.mean(axis=0)
        mean_dist = np.linalg.norm(points - mean, axis=1).mean()
        if mean_dist < 1e-8:
            raise ValueError('Corner pixels are degenerate')
        s = np.sqrt(2) / mean_dist
        t = np.array([[s, 0, -s*mean[0]], [0, s, -s*mean[1]], [0, 0, 1.]])
        q = (t @ np.c_[points, np.ones(len(points))].T).T
        return q[:, :2], t
    a, ts = norm(source)
    b, td = norm(target)
    rows = []
    for (x, y), (X, Y) in zip(a, b):
        rows.append([-x, -y, -1, 0, 0, 0, X*x, X*y, X])
        rows.append([0, 0, 0, -x, -y, -1, Y*x, Y*y, Y])
    _, sig, vt = np.linalg.svd(np.asarray(rows, dtype=float))
    if sig[-2] < 1e-10:
        raise ValueError('Corner configuration appears degenerate')
    h = vt[-1].reshape(3, 3)
    h = np.linalg.inv(td) @ h @ ts
    if abs(np.linalg.det(h)) < 1e-12:
        raise ValueError('Non-invertible homography')
    return h / h[2, 2]

def apply_h(points, h):
    p = np.asarray(points, dtype=float).reshape(-1, 2)
    z = (h @ np.c_[p, np.ones(len(p))].T).T
    if np.any(np.abs(z[:,2]) < 1e-12):
        raise ValueError('Invalid homography projection')
    return z[:,:2] / z[:,2:3]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='lid_geometry_config.json')
    p.add_argument('--corners', required=True, help='JSON file with measured corner pixels')
    p.add_argument('--output', default='lid_homography.json')
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text())
    measured = json.loads(Path(args.corners).read_text())
    px = measured['pixel_corners']
    if any(px.get(k) is None for k in KEYS):
        raise ValueError('Enter real [u,v] pixel locations for all four named corners')
    src = np.array([px[k] for k in KEYS], dtype=float)
    dst = np.array([config['nominal_corners_xy_stl_mm'][k] for k in KEYS], dtype=float)
    if src.shape != (4,2) or not np.isfinite(src).all():
        raise ValueError('pixel_corners must contain four finite [u,v] points')
    h = normalized_dlt(src, dst)
    invh = np.linalg.inv(h)
    invh /= invh[2,2]
    pred_mm = apply_h(src, h)
    corner_rms_mm = float(np.sqrt(np.mean(np.sum((pred_mm-dst)**2,axis=1))))
    # Corner RMS is almost always 0 for 4 correspondences; do not use as validation accuracy!
    holes = {name: {
          'center_lid_mm': vals['center_stl_mm'],
          'predicted_center_pixel': apply_h([vals['center_stl_mm']], invh)[0].tolist(),
    } for name, vals in config['holes'].items()}
    independent_errors = {}
    for name, observed in measured.get('observed_hole_pixels_optional', {}).items():
        if name not in config['holes']:
            raise ValueError('Unknown observed hole: '+name)
        predicted = apply_h([observed],h)[0]
        truth = np.array(config['holes'][name]['center_stl_mm'])
        independent_errors[name] = float(np.linalg.norm(predicted-truth))
    output = {
      'H_pixel_to_stl_xy_mm':h.tolist(),
      'H_stl_xy_mm_to_pixel':invh.tolist(),
      'measured_pixel_corners':{k:px[k] for k in KEYS},
      'corner_reprojection_rms_mm_not_independent_validation':corner_rms_mm,
      'independent_hole_center_errors_mm_if_provided':independent_errors,
      'predicted_hole_centers':holes,
      'warning':'For an actual 3D robot pose you ALSO need lid-to-world/robot transform, correct Z and object orientation. H alone does not solve insertion reachability.'
    }
    Path(args.output).write_text(json.dumps(output,indent=2))
    print('Wrote:',args.output)
    print('H pixel -> lid XY (mm):\n',np.array2string(h,precision=8))
    for name, val in holes.items():
        print(name, 'lid mm',val['center_lid_mm'],'estimated pixel',np.round(val['predicted_center_pixel'],2).tolist())
    print('WARNING: 4-point corner reprojection error is not an independent accuracy measurement.')

if __name__ == '__main__':
    main()
