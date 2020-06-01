import pytest
from numpy.testing import assert_allclose

from aspset510.geometry import roi_containing_points_2d


class TestRoiContainingPoints2d:
    @pytest.fixture
    def points_2d(self):
        return [
            [-5, 15],
            [10, 0],
            [5, 20],
        ]

    def test_smoke(self, points_2d):
        roi = roi_containing_points_2d(points_2d)
        assert isinstance(roi, tuple)
        assert len(roi) == 4

    def test_without_zoom(self, points_2d):
        roi = roi_containing_points_2d(points_2d)
        assert_allclose(roi, (-5, 0, 10, 20))

    def test_with_zoom(self, points_2d):
        roi = roi_containing_points_2d(points_2d, zoom=0.5)
        assert_allclose(roi, (-12.5, -10.0, 17.5, 30.0))
