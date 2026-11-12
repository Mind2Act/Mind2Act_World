import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mind2act_eval.adapters.coding_agent_kinematics import (
    X5Kinematics, _axis_matrix, _matrix_quaternion, _quaternion_matrix, _rpy_matrix,
)


def test_urdf_rpy_and_quaternion_math_against_independent_rotation_library():
    random = np.random.default_rng(5)
    for rpy in [np.zeros(3), np.array([np.pi, 0., 0.]), *random.uniform(-np.pi, np.pi, (30, 3))]:
        expected = Rotation.from_euler("xyz", rpy).as_matrix()
        np.testing.assert_allclose(_rpy_matrix(rpy), expected, atol=1e-14)
        quaternion = _matrix_quaternion(expected)
        np.testing.assert_allclose(_quaternion_matrix(quaternion), expected, atol=1e-14)
        assert abs(np.linalg.norm(quaternion) - 1.) < 1e-14
    np.testing.assert_allclose(_axis_matrix((0., 0., 1.), np.pi / 2), Rotation.from_euler("z", np.pi / 2).as_matrix(), atol=1e-14)


def test_fk_refuses_unpinned_robot_bytes(tmp_path):
    urdf = tmp_path / "other.urdf"
    urdf.write_text('<robot name="other"/>')
    with pytest.raises(ValueError, match="digest mismatch"):
        X5Kinematics(urdf, {})
