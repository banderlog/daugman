""" Code for `daugman_visual_explanation.ipynb`
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import itertools
import random

from daugman import daugman
from daugman import find_iris

from typing import List, Tuple, Iterable


class DaugmanVisualExplanation:
    def __init__(self, img_path: str, start_r=10, end_r=30, circle_step=2, points_step=3):
        self.img = self._get_new_image(img_path)
        self.start_r = start_r
        self.end_r = end_r
        self.circle_step = circle_step
        self.points_step = points_step
        self.all_points = self._get_all_potential_iris_centers(self.img)
        self.colors = self._get_unique_color_for_each_point(self.all_points)

    def _get_new_image(self, img_path, gray=False) -> np.ndarray:
        """ Get properly cropped BGR image, which looks like grayscale
        """
        img = cv2.imread(img_path)
        img = img[20:130, 20:130]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if not gray:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    def _get_all_potential_iris_centers(self, img: np.ndarray) -> List[Tuple[int, int]]:
        # get all potential points for search (from `find_iris()`)
        h = img.shape[0]
        # we will look only on dots within central 1/3 of image
        single_axis_range = range(int(h / 3), h - int(h / 3), self.points_step)
        all_points = list(itertools.product(single_axis_range, single_axis_range))
        return all_points

    def _get_unique_color_for_each_point(self, all_points: List[Tuple[int, int]]) -> List[Tuple[int, int, int]]:
        colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for i in self.all_points]
        return colors

    def plot_all_potential_iris_centers(self) -> np.ndarray:
        # plot all potential points
        img_dot = self.img.copy()
        for point, color in zip(self.all_points, self.colors):
            cv2.circle(img_dot, point, 0, color, -1)

        _ = plt.imshow(img_dot[::, ::, ::-1])
        return img_dot

    def plot_circles_for_one_center(self, img_dot: np.ndarray, dot_idx=0) -> np.ndarray:
        img_circles = img_dot.copy()

        # within circles in radii range from 10px to 1/4 of image side

        # plot the chosen potential point
        cv2.circle(img_circles, list(self.all_points)[dot_idx], 0, self.colors[dot_idx], 1)
        # plot all circle candidates for the single potential point
        img_circles = self._draw_circles(img_circles, self.all_points[dot_idx], self.colors[dot_idx],
                                         start_r=self.start_r, end_r=self.end_r, step=self.circle_step)

        _ = plt.imshow(img_circles[::, ::, ::-1])
        return img_circles

    def _draw_circles(self, img: np.ndarray,
                      center: Tuple[int, int], color: Tuple[int, int, int],
                      start_r: int, end_r: int, step: int,
                      alpha=0.5) -> np.ndarray:
        """ Part of ``daugman()`` modified for presentation purposes
        """
        # get separate coordinates
        x, y = center
        overlay = img.copy()

        radii = list(range(start_r, end_r, step))
        for r in radii:
            cv2.circle(overlay, center, r, color, 1)

        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        return img

    def plot_best_circle_for_single_potential_iris_center(self, img_dot: np.ndarray,
                                                          dot_idx: int, color=None, alpha=0.8) -> np.ndarray:
        gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        # get best circle
        _, best_radius = daugman(gray_img, self.all_points[dot_idx],
                                 self.start_r, self.end_r, self.circle_step)
        # plot best circle
        if not color:
            color = self.colors[dot_idx]

        overlay = img_dot.copy()
        cv2.circle(overlay, self.all_points[dot_idx], best_radius, color, 1)
        img_dot = cv2.addWeighted(overlay, alpha, img_dot, 1 - alpha, 0)
        return img_dot

    def plot_best_circle_for_a_few_potential_iris_centers(self, img_dot: np.ndarray,
                                                          idxs: Iterable[int]) -> np.ndarray:
        img = img_dot.copy()

        for idx in idxs:
            img = self.plot_best_circle_for_single_potential_iris_center(img, idx)

        _ = plt.imshow(img[::, ::, ::-1])
        return img_dot

    def find_iris(self) -> np.ndarray:
        gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        answer = find_iris(gray_img, daugman_start=self.start_r, daugman_end=self.end_r,
                           daugman_step=self.circle_step, points_step=self.points_step)
        iris_center, iris_rad = answer
        out = self.img.copy()
        cv2.circle(out, iris_center, iris_rad, (0, 0, 255), 1)
        _ = plt.imshow(out[::, ::, ::-1])
        return out

    def plot_pixel_intensity_delta_pic(self) -> None:
        # white image
        img = np.full([100, 100, 3], 255, dtype=np.uint8)
        # black circle
        img = cv2.circle(img, (50, 50), 20, [0, 0, 0], -1)
        # yellow
        img = cv2.circle(img, (50, 50), 10, [255, 255, 0], 1)
        # green
        img = cv2.circle(img, (50, 50), 15, [0, 255, 0], 1)
        # red
        img = cv2.circle(img, (50, 50), 20, [255, 0, 0], 1)
        # blue
        img = cv2.circle(img, (50, 50), 25, [0, 0, 255], 1)
        _ = plt.imshow(img)
