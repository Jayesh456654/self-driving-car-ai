"""
geometry.py - Math utilities for 2D line segment intersection and vector operations.
"""
import math


def line_intersection(p1, p2, p3, p4):
    """
    Computes the intersection point between line segment p1-p2 and p3-p4.
    Returns (ix, iy, distance_from_p1) if they intersect, else None.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-9:
        return None

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / denom

    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        ix = x1 + t * (x2 - x1)
        iy = y1 + t * (y2 - y1)
        dist = math.hypot(ix - x1, iy - y1)
        return (ix, iy, dist)

    return None


def distance(p1, p2):
    """Euclidean distance between two points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def rotate_point(origin, point, angle_rad):
    """Rotate a point counterclockwise by angle_rad around origin."""
    ox, oy = origin
    px, py = point
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    qx = ox + cos_a * (px - ox) - sin_a * (py - oy)
    qy = oy + sin_a * (px - ox) + cos_a * (py - oy)
    return (qx, qy)
