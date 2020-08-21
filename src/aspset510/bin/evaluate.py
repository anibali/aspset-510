"""Evaluate the accuracy of predictions on the ASPset-510 dataset.
"""

import argparse
import sys
from pathlib import Path

from posekit.skeleton import skeleton_registry, skeleton_converter
from tqdm import tqdm

from aspset510 import Aspset510
from aspset510.evaluation import Joints3dEvaluator, find_and_load_prediction
from aspset510.scale import to_univ_scale
from aspset510.util import add_boolean_argument


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True,
                        help='path to the base dataset directory')
    parser.add_argument('--predictions', type=str, required=True,
                        help='path to the predictions directory')
    parser.add_argument('--split', type=str, required=True,
                        help='split of the dataset to evaluate on (e.g. train, val, or test)')
    add_boolean_argument(parser, 'univ', description='universal pose scale', default=False)
    add_boolean_argument(parser, 'skip_missing', description='skip missing prediction files',
                         default=False)
    return parser


def get_joint_positions(mocap, sample_rate):
    factor = mocap.sample_rate / sample_rate
    int_factor = int(factor)
    if int_factor != factor:
        raise ValueError('sample rate is not evenly divisible')
    return mocap.joint_positions[::int_factor]


def main(args):
    opts = argument_parser().parse_args(args)

    aspset = Aspset510(opts.data_dir)
    clips = aspset.clips(opts.split)
    preds_dir = Path(opts.predictions)

    # Skeleton to use for evaluation. Joints from predictions and ground truth will be converted
    # to this skeleton representation.
    skeleton = skeleton_registry['aspset_17j']

    evaluator = Joints3dEvaluator(skeleton)

    if opts.split == 'test':
        # Test set evaluation is performed at 10 frames per second.
        sample_rate = 10
    else:
        # Evaluation on training/validation data is performed at the full 50 frames per second.
        sample_rate = 50

    n_prediction_files = 0

    for clip in tqdm(clips, leave=True, ascii=True):
        # Find which camera angles are to be used for evaluating this clip.
        camera_ids = aspset.cameras_for_clip[(clip.subject_id, clip.clip_id)]
        # Prediction files don't need to specify the camera ID when there is only one possibility.
        include_unknown_camera = len(camera_ids) == 1

        # Load and prepare the ground truth 3D joint annotations.
        gt_mocap = clip.load_mocap()
        gt_skeleton = skeleton_registry[gt_mocap.skeleton_name]
        gt_joints_3d = get_joint_positions(gt_mocap, sample_rate)
        gt_joints_3d = skeleton_converter.convert(gt_joints_3d, gt_skeleton, skeleton)
        if opts.univ:
            gt_joints_3d = to_univ_scale(gt_joints_3d, skeleton)

        # Load and add predictions to the evaluation.
        for camera_id in camera_ids:
            try:
                pred_mocap = find_and_load_prediction(preds_dir, clip.subject_id, clip.clip_id,
                                                      camera_id, include_unknown_camera)
            except:
                if opts.skip_missing:
                    continue
                raise
            pred_skeleton = skeleton_registry[pred_mocap.skeleton_name]
            pred_joints_3d = skeleton_converter.convert(get_joint_positions(pred_mocap, sample_rate),
                                                        pred_skeleton, skeleton)
            evaluator.add(pred_joints_3d, gt_joints_3d)
            n_prediction_files += 1

    print(f'Found {len(evaluator)} poses in {n_prediction_files} prediction files.\n')

    # Print the evaluation results to stdout.
    evaluator.print_results()


if __name__ == '__main__':
    main(sys.argv[1:])
