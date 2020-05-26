from glupy.math import ensure_homogeneous


class Camera:
    def __init__(self, intrinsic_matrix, extrinsic_matrix):
        self.intrinsic_matrix = intrinsic_matrix
        self.extrinsic_matrix = extrinsic_matrix

    @property
    def projection_matrix(self):
        return self.intrinsic_matrix @ self.extrinsic_matrix

    def world_to_camera_space(self, points_3d):
        """Transform points from 3D world space to 3D camera space.

        Args:
            points_3d: 3D points in world space.

        Returns:
            Homogeneous 3D points in camera space.
        """
        points_3d = ensure_homogeneous(points_3d, d=3)
        return points_3d @ self.extrinsic_matrix.T

    def camera_to_image_space(self, points_3d):
        """Transform points from 3D camera space to 2D image space.

        Args:
            points_3d: 3D points in camera space.

        Returns:
            Homogeneous 2D points in image space.
        """
        points_3d = ensure_homogeneous(points_3d, d=3)
        return points_3d @ self.intrinsic_matrix.T

    def world_to_image_space(self, points_3d):
        """Transform points from 3D world space to 2D image space.

        Args:
            points_3d: 3D points in world space.

        Returns:
            Homogeneous 2D points in image space.
        """
        points_3d = ensure_homogeneous(points_3d, d=3)
        return points_3d @ self.projection_matrix.T
