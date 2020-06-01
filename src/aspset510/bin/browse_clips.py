"""Show a GUI for browsing clips from the ASPset-510 dataset.
"""

import argparse
import sys
import tkinter as tk
from tkinter import ttk

import cv2
import matplotlib.pyplot as plt
import numpy as np
from aspset510.geometry import roi_containing_points_2d, square_containing_rectangle
from glupy.math import to_cartesian
from matplotlib.axes import Axes
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from posekit.skeleton import skeleton_registry

from aspset510 import Aspset510
from aspset510.plot import plot_joints_2d, plot_joints_3d


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True,
                        help='path to the base dataset directory')
    return parser


class DatasetBrowser(tk.Tk):
    # Since the videos are 4K, we'll scale them on load for faster GUI performance.
    IMAGE_SCALE = 1 / 2

    def __init__(self, aspset: Aspset510):
        super().__init__()
        self.geometry('1280x800')
        self.wm_title(f'ASPset-510 dataset browser')
        self.protocol('WM_DELETE_WINDOW', lambda: self.quit())

        self.aspset = aspset

        self.var_split = tk.StringVar()
        self.var_subject = tk.StringVar()
        self.var_clip = tk.StringVar()
        self.var_camera = tk.StringVar(value=aspset.CAMERA_IDS[0])
        self.var_frame = tk.StringVar()
        self.var_zoom = tk.IntVar(value=1)

        self.var_split.trace('w', self.on_split_change)
        self.var_subject.trace('w', self.on_subject_change)
        self.var_clip.trace('w', self.on_clip_change)
        self.var_camera.trace('w', self._trigger_refresh)
        self.var_frame.trace('w', self._trigger_refresh)
        self.var_zoom.trace('w', self._trigger_refresh)

        toolbar = self._create_toolbar()
        toolbar.pack(side=tk.TOP, fill=tk.X)

        fig, self.canvas = self._create_figure()
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=1)

        gs = GridSpec(1, 2, figure=fig)
        self.ax_image: Axes = fig.add_subplot(gs[0, 0])
        self.ax_joints_3d: Axes3D = fig.add_subplot(gs[0, 1], projection='3d')

        fig.subplots_adjust(0.01, 0.01, 0.99, 0.99, 0.05, 0.05)

        # Set initial state of comboboxes (will result in an example being shown).
        self.cmb_split.current(0)

    def _create_toolbar(self):
        toolbar = tk.Frame(self, bd=1, relief=tk.RAISED)

        cmb_split = ttk.Combobox(toolbar, textvariable=self.var_split, state='readonly')
        cmb_split['values'] = self.aspset.split_names
        cmb_split.pack(side=tk.LEFT, fill=tk.Y, padx=2, pady=2)
        self.cmb_split = cmb_split

        cmb_subject = ttk.Combobox(toolbar, textvariable=self.var_subject, state='readonly')
        cmb_subject.pack(side=tk.LEFT, fill=tk.Y, padx=2, pady=2)
        self.cmb_subject = cmb_subject

        cmb_clip = ttk.Combobox(toolbar, textvariable=self.var_clip, state='readonly')
        cmb_clip.pack(side=tk.LEFT, fill=tk.Y, padx=2, pady=2)
        self.cmb_clip = cmb_clip

        cmb_camera = ttk.Combobox(toolbar, textvariable=self.var_camera, state='readonly')
        cmb_camera['values'] = self.aspset.CAMERA_IDS
        cmb_camera.pack(side=tk.LEFT, fill=tk.Y, padx=2, pady=2)
        self.cmb_camera = cmb_camera

        spn_frame = tk.Spinbox(toolbar, textvariable=self.var_frame, wrap=True)
        spn_frame.pack(side=tk.LEFT, fill=tk.Y, padx=2, pady=2)
        self.spn_frame = spn_frame

        chk_zoom = tk.Checkbutton(toolbar, text='Crop image', variable=self.var_zoom)
        chk_zoom.pack(side=tk.LEFT, fill=tk.Y, padx=2, pady=2)
        self.chk_zoom = chk_zoom

        return toolbar

    def _create_figure(self):
        fig: Figure = plt.figure()
        canvas = FigureCanvasTkAgg(fig, master=self)
        NavigationToolbar2Tk(canvas, self)
        return fig, canvas

    def on_split_change(self, *args):
        split = self.var_split.get()
        subjects = set()
        for clip_subject_id, _ in self.aspset.splits[split]:
            subjects.add(clip_subject_id)
        subjects = list(sorted(subjects))
        self.cmb_subject['values'] = subjects
        self.cmb_subject.current(0)

    def on_subject_change(self, *args):
        split = self.var_split.get()
        subject_id = self.var_subject.get()
        clips = set()
        for clip_subject_id, clip_id in self.aspset.splits[split]:
            if clip_subject_id == subject_id:
                clips.add(clip_id)
        clips = list(sorted(clips))
        self.cmb_clip['values'] = clips
        self.cmb_clip.current(0)

    def on_clip_change(self, *args):
        subject_id = self.var_subject.get()
        clip_id = self.var_clip.get()
        clip = self.aspset.clip(subject_id, clip_id)
        self.spn_frame.config(from_=0, to=clip.frame_count - 1)
        self.var_frame.set('0')

    def _trigger_refresh(self, *args):
        self.refresh()

    def run(self):
        self.mainloop()

    def refresh(self):
        subject_id = self.var_subject.get()
        clip_id = self.var_clip.get()
        clip = self.aspset.clip(subject_id, clip_id)
        try:
            frame = int(self.var_frame.get())
        except ValueError:
            frame = 0
        camera_id = self.var_camera.get()

        video_path = clip.get_video_path(camera_id)
        video = cv2.VideoCapture(str(video_path))
        video.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, image = video.read()
        if not ret:
            raise RuntimeError(f'failed to read frame from video: {video_path}')
        height, width = image.shape[:2]
        image = cv2.resize(image, (int(width * self.IMAGE_SCALE), int(height * self.IMAGE_SCALE)))
        image = np.flip(np.asarray(image), -1)

        camera = clip.load_camera(camera_id)
        mocap = clip.load_mocap()
        skeleton = skeleton_registry[mocap.skeleton_name]
        joints_3d = mocap.joint_positions
        joints_2d = to_cartesian(camera.world_to_image_space(joints_3d)) * self.IMAGE_SCALE

        self.ax_image.cla()
        self.ax_image.imshow(image)
        self.ax_image.set_xticks([])
        self.ax_image.set_yticks([])
        plot_joints_2d(self.ax_image, joints_2d[frame], skeleton)

        if self.var_zoom.get():
            x1, y1, x2, y2 = square_containing_rectangle(
                *roi_containing_points_2d(joints_2d[frame], zoom=2/3)
            )
            self.ax_image.set_xlim(x1, x2)
            self.ax_image.set_ylim(y2, y1)

        elev = self.ax_joints_3d.elev
        azim = self.ax_joints_3d.azim
        self.ax_joints_3d.cla()
        plot_joints_3d(self.ax_joints_3d, joints_3d[frame], skeleton)
        # Restore the previous viewing angle.
        self.ax_joints_3d.view_init(elev, azim)

        self.canvas.draw()


def main(args):
    opts = argument_parser().parse_args(args)
    aspset = Aspset510(opts.data_dir)
    DatasetBrowser(aspset).run()


if __name__ == '__main__':
    main(sys.argv[1:])
