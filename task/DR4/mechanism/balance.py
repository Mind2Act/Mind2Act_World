"""Original per-ball tray-frame displacement; no historical scoring formula."""
import numpy as np
NUMERIC_TOLERANCE_M = .0005


def local_positions(tray_position, rotation, balls):
    return (np.asarray(balls) - np.asarray(tray_position)) @ np.asarray(rotation)


def displacement(local, anchors):
    return np.linalg.norm(np.asarray(local)[:, :2] - np.asarray(anchors)[:, :2], axis=1)
