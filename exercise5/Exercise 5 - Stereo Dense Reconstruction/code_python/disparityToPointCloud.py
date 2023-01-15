import numpy as np


def disparityToPointCloud(disp_img, K, baseline, left_img):
    """
    points should be Nx3 and intensities N, where N is the amount of pixels which have a valid disparity.
    I.e., only return points and intensities for pixels of left_img which have a valid disparity estimate!
    The i-th intensity should correspond to the i-th point.
    """

