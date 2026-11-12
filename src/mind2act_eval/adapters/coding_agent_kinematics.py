"""Public official X5 URDF forward kinematics from measured joint radians."""
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

URDF_SHA256 = "7aaa308a1a5b6a213d914db86ba09ddd6252f25cb498ac9a148bfa6c7166a423"


def _axis_matrix(axis, angle):
    x, y, z = axis
    cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + np.sin(angle) * cross + (1. - np.cos(angle)) * (cross @ cross)


def _rpy_matrix(rpy):
    roll, pitch, yaw = rpy
    return (_axis_matrix((0., 0., 1.), yaw) @ _axis_matrix((0., 1., 0.), pitch)
            @ _axis_matrix((1., 0., 0.), roll))


def _quaternion_matrix(quaternion):
    w, x, y, z = quaternion
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def _matrix_quaternion(matrix):
    # Symmetric eigenproblem is stable at 180 degrees as well as identity.
    m = matrix
    k = np.array([[m[0,0]-m[1,1]-m[2,2], m[1,0]+m[0,1], m[2,0]+m[0,2], m[2,1]-m[1,2]],
                  [m[1,0]+m[0,1], m[1,1]-m[0,0]-m[2,2], m[2,1]+m[1,2], m[0,2]-m[2,0]],
                  [m[2,0]+m[0,2], m[2,1]+m[1,2], m[2,2]-m[0,0]-m[1,1], m[1,0]-m[0,1]],
                  [m[2,1]-m[1,2], m[0,2]-m[2,0], m[1,0]-m[0,1], m.trace()]]) / 3.
    _, vectors = np.linalg.eigh(k)
    quaternion = vectors[:, -1][[3, 0, 1, 2]]
    return quaternion if quaternion[0] >= 0 else -quaternion


class X5Kinematics:
    def __init__(self, urdf, mounts):
        raw = Path(urdf).read_bytes()
        if hashlib.sha256(raw).hexdigest() != URDF_SHA256:
            raise ValueError("official public X5 URDF digest mismatch")
        tree = ET.fromstring(raw)
        self.joints = []
        parent = "base_link"
        for index in range(1, 7):
            joint = tree.find(f"joint[@name='joint{index}']")
            if joint is None or joint.attrib["type"] != "revolute" or joint.find("parent").attrib["link"] != parent:
                raise ValueError("official X5 arm chain mismatch")
            parent = joint.find("child").attrib["link"]
            origin = joint.find("origin")
            xyz = np.fromstring(origin.attrib["xyz"], sep=" ")
            rpy = np.fromstring(origin.attrib["rpy"], sep=" ")
            axis = np.fromstring(joint.find("axis").attrib["xyz"], sep=" ")
            if xyz.shape != (3,) or rpy.shape != (3,) or axis.shape != (3,) or np.linalg.norm(axis) == 0:
                raise ValueError("invalid X5 joint transform")
            transform = np.eye(4)
            transform[:3, :3] = _rpy_matrix(rpy)
            transform[:3, 3] = xyz
            self.joints.append((transform, axis / np.linalg.norm(axis)))
        if parent != "link6":
            raise ValueError("native flange must be the official link6 frame")
        if not isinstance(mounts, dict) or set(mounts) != {"left", "right"}:
            raise ValueError("explicit left/right public world mounts required")
        self.mounts = {}
        for arm, value in mounts.items():
            pose = np.asarray(value, dtype=float)
            if pose.shape != (7,) or not np.isfinite(pose).all() or abs(np.linalg.norm(pose[3:]) - 1) > 1e-6:
                raise ValueError("public mount must be xyz + unit wxyz quaternion")
            transform = np.eye(4)
            transform[:3, :3] = _quaternion_matrix(pose[3:])
            transform[:3, 3] = pose[:3]
            self.mounts[arm] = transform

    def flange_pose(self, arm, radians):
        q = np.asarray(radians, dtype=float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError("FK requires six finite measured radians")
        transform = self.mounts[arm].copy()
        for angle, (origin, axis) in zip(q, self.joints):
            motion = np.eye(4)
            motion[:3, :3] = _axis_matrix(axis, angle)
            transform = transform @ origin @ motion
        quaternion = _matrix_quaternion(transform[:3, :3])
        return np.r_[transform[:3, 3], quaternion]
