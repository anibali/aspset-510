"""Evaluate the accuracy of predictions on the ASPset-510 dataset.
"""

import argparse
import sys

from tqdm import tqdm

from aspset510 import Aspset510
from aspset510.evaluation import Joints3dEvaluator, EvaluationDataLoader
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


def main(args):
    opts = argument_parser().parse_args(args)

    eval_data = EvaluationDataLoader(
        Aspset510(opts.data_dir),
        opts.predictions,
        opts.split,
        'aspset_17j',
        opts.univ,
        opts.skip_missing,
    )
    evaluator = Joints3dEvaluator(eval_data.skeleton)
    n_prediction_files = 0

    for pred_joints_3d_by_camera, gt_joints_3d in tqdm(eval_data, leave=True, ascii=True):
        for pred_joints_3d in pred_joints_3d_by_camera.values():
            evaluator.add(pred_joints_3d, gt_joints_3d)
            n_prediction_files += 1

    # Print the evaluation results to stdout.
    print(f'Found {len(evaluator)} poses in {n_prediction_files} prediction files.\n')
    evaluator.print_results()


if __name__ == '__main__':
    main(sys.argv[1:])
