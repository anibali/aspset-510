from pathlib import Path

import numpy as np
from posekit.io import load_mocap, Mocap
from posekit.skeleton import Skeleton, skeleton_registry, skeleton_converter
from posekit.skeleton.utils import assert_plausible_skeleton, procrustes, absolute_to_root_relative

from aspset510 import Aspset510
from aspset510.scale import to_univ_scale
from aspset510.util import FSPath


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

    def __len__(self):
        return len(self.expected_poses)

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


def find_and_load_prediction(
    preds_dir: FSPath,
    subject_id: str,
    clip_id: str,
    camera_id: str,
    include_unknown_camera: bool = False,
) -> Mocap:
    """Find and load specific motion capture data from a directory containing prediction files.

    Args:
        preds_dir: The root predictions directory.
        subject_id: The subject ID of the prediction file to find.
        clip_id: The clip ID of the prediction file to find.
        camera_id: The camera ID of the prediction file to find.
        include_unknown_camera: If ``True``, prediction files without a camera ID are assumed to
            be for ``camera_id``.

    Returns:
        Motion capture data for the specified clip and camera.
    """
    preds_dir = Path(preds_dir)
    pred_files = list(preds_dir.rglob(f'{subject_id}-{clip_id}-{camera_id}.*'))
    if include_unknown_camera:
        pred_files.extend(preds_dir.rglob(f'{subject_id}-{clip_id}.*'))
    if len(pred_files) == 0:
        raise RuntimeError(f'no prediction file found for {subject_id}-{clip_id}-{camera_id}')
    if len(pred_files) > 1:
        raise RuntimeError(f'multiple prediction files found for {subject_id}-{clip_id}-{camera_id}')
    return load_mocap(pred_files[0])


def get_joint_positions(mocap, sample_rate):
    factor = mocap.sample_rate / sample_rate
    int_factor = int(factor)
    if int_factor != factor:
        raise ValueError('sample rate is not evenly divisible')
    return mocap.joint_positions[::int_factor]


class EvaluationDataLoader:
    def __init__(
        self,
        aspset: Aspset510,
        preds_dir: FSPath,
        split: str,
        skeleton_name: str = 'aspset_17j',
        univ: bool = False,
        skip_missing: bool = False,
    ):
        """Create an iterable for loading predicted and corresponding ground truth 3D poses.

        Args:
            aspset:
            preds_dir: The root predictions directory.
            split: The dataset split to load data from (e.g. 'train', 'val', 'test').
            skeleton_name: Skeleton to use for evaluation. Joints from predictions and ground truth
                will be converted to this skeleton representation.
            univ: If ``True``, ground truth poses will be converted to universal scale.
            skip_missing: If ``True``, missing predictions will be skipped instead of raising an
                exception.
        """
        self.aspset = aspset
        self.preds_dir = Path(preds_dir)
        self.split = split
        self.skeleton = skeleton_registry[skeleton_name]
        self.univ = univ
        self.skip_missing = skip_missing

        if self.split == 'test':
            # Test set evaluation is performed at 10 frames per second.
            self.sample_rate = 10
        else:
            # Evaluation on training/validation data is performed at the full 50 frames per second.
            self.sample_rate = 50

        self.clips = aspset.clips(self.split)

    def __len__(self):
        return len(self.clips)

    def __iter__(self):
        for clip in self.clips:
            # Find which camera angles are to be used for evaluating this clip.
            camera_ids = self.aspset.cameras_for_clip[(clip.subject_id, clip.clip_id)]
            # Prediction files don't need to specify the camera ID when there is only one possibility.
            include_unknown_camera = len(camera_ids) == 1

            # Load and prepare the ground truth 3D joint annotations.
            gt_mocap = clip.load_mocap()
            gt_skeleton = skeleton_registry[gt_mocap.skeleton_name]
            gt_joints_3d = get_joint_positions(gt_mocap, self.sample_rate)
            gt_joints_3d = skeleton_converter.convert(gt_joints_3d, gt_skeleton, self.skeleton)
            if self.univ:
                gt_joints_3d = to_univ_scale(gt_joints_3d, self.skeleton)

            # Load predictions.
            pred_joints_3d_by_camera = {}
            for camera_id in camera_ids:
                try:
                    pred_mocap = find_and_load_prediction(self.preds_dir, clip.subject_id,
                                                          clip.clip_id, camera_id,
                                                          include_unknown_camera)
                except:
                    if self.skip_missing:
                        continue
                    else:
                        raise
                pred_skeleton = skeleton_registry[pred_mocap.skeleton_name]
                pred_joints_3d = get_joint_positions(pred_mocap, self.sample_rate)
                pred_joints_3d = skeleton_converter.convert(pred_joints_3d, pred_skeleton, self.skeleton)
                pred_joints_3d_by_camera[camera_id] = pred_joints_3d

            # Yield predictions for each camera and the ground truth.
            yield pred_joints_3d_by_camera, gt_joints_3d
