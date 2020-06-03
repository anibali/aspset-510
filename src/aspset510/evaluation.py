import numpy as np
from posekit.skeleton import Skeleton
from posekit.skeleton.utils import assert_plausible_skeleton, procrustes, absolute_to_root_relative


def calculate_mpjpe(actual_poses, expected_poses):
    """Calculate the average mean per joint position error."""
    total_mpjpe = 0
    n_total = 0
    for actual, expected in zip(actual_poses, expected_poses):
        # Calculate PCK accuracy for this pose.
        dists = np.linalg.norm(expected - actual, ord=2, axis=-1)
        # Add accuracy to running total.
        total_mpjpe += dists.mean()
        n_total += 1
    # Return 0 if we have no per-pose accuracies.
    if n_total == 0:
        return 0
    # Calculate average accuracy across all poses.
    return float(total_mpjpe / n_total)


def calculate_pck(actual_poses, expected_poses, threshold=150):
    """Calculate the average percentage of correct keypoints."""
    total_accuracy = 0
    n_total = 0
    for actual, expected in zip(actual_poses, expected_poses):
        # Calculate PCK accuracy for this pose.
        dists = np.linalg.norm(expected - actual, ord=2, axis=-1)
        accuracy = (dists <= threshold).mean()
        # Add accuracy to running total.
        total_accuracy += accuracy
        n_total += 1
    # Return 0 if we have no per-pose accuracies.
    if n_total == 0:
        return 0
    # Calculate average accuracy across all poses.
    return float(total_accuracy / n_total)


class Joints3dEvaluator:
    def __init__(self, skeleton: Skeleton):
        self.skeleton = skeleton
        self.actual_poses = []
        self.pa_actual_poses = []
        self.expected_poses = []

    def add(self, actual, expected):
        actual = np.asarray(actual, np.float32)
        expected = np.asarray(expected, np.float32)
        assert_plausible_skeleton(actual, self.skeleton)
        assert_plausible_skeleton(expected, self.skeleton)
        assert actual.shape == expected.shape
        if actual.ndim > 2:
            for actual_part, expected_part in zip(actual, expected):
                self.add(actual_part, expected_part)
            return
        self.actual_poses.append(actual)
        self.pa_actual_poses.append(procrustes(expected, actual))
        self.expected_poses.append(expected)

    def _to_rr(self, poses):
        """Convert poses to root-relative joint locations."""
        return [absolute_to_root_relative(pose, self.skeleton) for pose in poses]

    def mpjpe(self):
        """Calculate the average mean per joint position error."""
        return calculate_mpjpe(self.actual_poses, self.expected_poses)

    def rr_mpjpe(self):
        """Calculate the root-relative average mean per joint position error."""
        return calculate_mpjpe(self._to_rr(self.actual_poses), self._to_rr(self.expected_poses))

    def pa_mpjpe(self):
        """Calculate the procrustes-aligned average mean per joint position error."""
        return calculate_mpjpe(self.pa_actual_poses, self.expected_poses)

    def pck(self, threshold=150):
        """Calculate the average percentage of correct keypoints."""
        return calculate_pck(self.actual_poses, self.expected_poses, threshold=threshold)

    def rr_pck(self, threshold=150):
        """Calculate the root-relative average percentage of correct keypoints."""
        return calculate_pck(self._to_rr(self.actual_poses), self._to_rr(self.expected_poses),
                             threshold=threshold)

    def pa_pck(self, threshold=150):
        """Calculate the procrustes-aligned average percentage of correct keypoints."""
        return calculate_pck(self.pa_actual_poses, self.expected_poses, threshold=threshold)

    def collect_results(self):
        return {
            'MPJPE': self.mpjpe(),
            'Root-relative MPJPE': self.rr_mpjpe(),
            'Procrustes-aligned MPJPE': self.pa_mpjpe(),
            'PCK': self.pck(),
            'Root-relative PCK': self.rr_pck(),
            'Procrustes-aligned PCK': self.pa_pck(),
        }

    def print_results(self):
        results = self.collect_results()
        for name, value in results.items():
            print(f'{name}: {value:0.4f}')