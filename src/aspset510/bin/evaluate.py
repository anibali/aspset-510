"""Evaluate the accuracy of predictions on the ASPset-510 dataset.
"""

import argparse
import sys
from pathlib import Path

from posekit.io import load_mocap
from posekit.skeleton import skeleton_registry, skeleton_converter
from tqdm import tqdm

from aspset510 import Aspset510
from aspset510.evaluation import Joints3dEvaluator
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
        sample_rate = 10
    else:
        sample_rate = 50

    for clip in tqdm(clips, leave=True, ascii=True):
        # FIXME: Support predictions for multiple camera angles (relevant for train and val splits).
        pred_files = list(preds_dir.rglob(f'{clip.subject_id}-{clip.clip_id}.*'))
        if len(pred_files) != 1:
            raise RuntimeError(f'no unique prediction file for {clip}')
        pred_mocap = load_mocap(pred_files[0])
        pred_skeleton = skeleton_registry[pred_mocap.skeleton_name]
        pred_joints_3d = skeleton_converter.convert(get_joint_positions(pred_mocap, sample_rate),
                                                    pred_skeleton, skeleton)
        gt_mocap = clip.load_mocap()
        gt_skeleton = skeleton_registry[gt_mocap.skeleton_name]
        gt_joints_3d = skeleton_converter.convert(get_joint_positions(gt_mocap, sample_rate),
                                                  gt_skeleton, skeleton)
        if opts.univ:
            gt_joints_3d = to_univ_scale(gt_joints_3d, skeleton)
        evaluator.add(pred_joints_3d, gt_joints_3d)

    evaluator.print_results()


if __name__ == '__main__':
    main(sys.argv[1:])
