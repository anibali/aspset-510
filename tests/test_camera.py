from glupy.math import to_cartesian
from numpy.testing import assert_allclose


def test_world_to_image_space(camera, joints_3d, joints_2d):
    actual = to_cartesian(camera.world_to_image_space(joints_3d))
    assert_allclose(actual, joints_2d)
