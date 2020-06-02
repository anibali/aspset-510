"""A simple script for visualising a static 3D pose from the ASPset-510 dataset.

Controls
========
Arrow Keys: Rotate the camera around the subject.
Page Up/Down: Move the camera towards/away from the subject.
Home: Reset the camera.
"""

import argparse
import sys

from glupy.math import ensure_cartesian
from posekit.gui.pose_viewer import PoseViewer
from posekit.skeleton import skeleton_registry

from aspset510 import Aspset510


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True,
                        help='path to the base dataset directory')
    parser.add_argument('--subject', type=str, required=True,
                        help='subject ID')
    parser.add_argument('--clip', type=str, required=True,
                        help='clip ID')
    parser.add_argument('--frame', type=int, default=0,
                        help='frame index')
    return parser


def main(args):
    opts = argument_parser().parse_args(args)

    aspset = Aspset510(opts.data_dir)
    clip = aspset.clip(opts.subject, opts.clip)
    mocap = clip.load_mocap()
    pose = ensure_cartesian(mocap.joint_positions[opts.frame], d=3)
    gui = PoseViewer(pose, skeleton_registry[mocap.skeleton_name])
    gui.run()


if __name__ == '__main__':
    main(sys.argv[1:])
