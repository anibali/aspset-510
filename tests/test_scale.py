import pytest
from posekit.skeleton import skeleton_registry
from posekit.skeleton.utils import calculate_knee_neck_height
from numpy.testing import assert_array_almost_equal, assert_allclose

from aspset510.scale import to_univ_scale, to_root_relative_univ_scale


def test_to_univ_scale(joints_3d):
    skeleton = skeleton_registry['aspset_17j']
    univ_joints_3d = to_univ_scale(joints_3d, skeleton)
    assert calculate_knee_neck_height(univ_joints_3d, skeleton.joint_names) == pytest.approx(910)
    # Assert that the joints were scaled about the origin.
    ratio = joints_3d / univ_joints_3d
    assert_allclose(ratio, ratio[0, 0])


def test_to_root_relative_univ_scale(joints_3d):
    skeleton = skeleton_registry['aspset_17j']
    univ_joints_3d = to_root_relative_univ_scale(joints_3d, skeleton)
    assert calculate_knee_neck_height(univ_joints_3d, skeleton.joint_names) == pytest.approx(910)
    # Assert that the root joint location has not changed.
    assert_array_almost_equal(univ_joints_3d[skeleton.root_joint_id], joints_3d[skeleton.root_joint_id])
