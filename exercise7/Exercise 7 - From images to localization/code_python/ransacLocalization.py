import cv2
import numpy as np

from code_previous_exercises.estimate_pose_dlt import estimatePoseDLT
from code_previous_exercises.projectPoints import projectPoints


def ransacLocalization(matched_query_keypoints, corresponding_landmarks, K):
    """
    best_inlier_mask should be 1xnum_matched and contain, only for the matched keypoints,
    False if the match is an outlier, True otherwise.
    """
