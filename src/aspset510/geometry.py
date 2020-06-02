from functools import reduce

import numpy as np
from glupy.math import mat3, ensure_homogeneous, ensure_cartesian


def transform_points_2d(points, T):
    """Apply an affine transformation to an array of 2D points."""
    points = ensure_homogeneous(points, d=2)
    return ensure_cartesian(points @ T.transpose(-1, -2), d=2)


def transform_points_3d(points, T):
    """Apply an affine transformation to an array of 3D points."""
    points = ensure_homogeneous(points, d=3)
    return ensure_cartesian(points @ T.transpose(-1, -2), d=3)


def zoom_roi(roi, zoom):
    if zoom == 1:
        return roi
    x1, y1, x2, y2 = roi
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    T = reduce(lambda A, B: np.dot(B, A), [
        mat3.translate(-cx, -cy),
        mat3.scale(1 / zoom),
        mat3.translate(cx, cy),
    ])
    corners = transform_points_2d(np.asarray([[x1, y1, 1], [x2, y2, 1]], dtype=float), T)
    return (corners[0, 0], corners[0, 1], corners[1, 0], corners[1, 1])


def roi_containing_points_2d(points, zoom=1):
    """Find the axis-aligned bounding box containing a set of 2D points.

    Args:
        points: The points to be contained in the box.
        zoom (float): Factor by which to scale the box. Values less than 1 will expand the box.
                      Default is 1 (no zoom).

    Returns:
        A four-element tuple (x1, y1, x2, y2) representing the bounding box.
    """
    assert zoom >= 0
    points = ensure_cartesian(np.asarray(points), d=2)
    x1, y1 = points.reshape(-1, 2).min(0)
    x2, y2 = points.reshape(-1, 2).max(0)
    roi = zoom_roi((x1, y1, x2, y2), zoom)
    return roi


def square_containing_rectangle(rect):
    x1, y1, x2, y2 = rect
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    s = max(x2 - x1, y2 - y1) / 2
    return (cx - s, cy - s, cx + s, cy + s)
