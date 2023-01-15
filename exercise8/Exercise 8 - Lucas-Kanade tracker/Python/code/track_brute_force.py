import numpy as np

from get_sim_warp import getSimWarp
from get_warped_patch import getWarpedPatch

def trackBruteForce(I_R, I, x_T, r_T, r_D):
    """
    Input
        I_R     np.ndarray reference image
        I       np.ndarray image to track points in
        x_T     1 x 2, point to track
        r_T     scalar, radius of the patch size to track
        r_D     scalar, radius of the patch to search within for best dx
    Output
        dx      1 x 2, translation that best explains where x_T is in
                image I
        ssds    SSDs for all values of dx within the patch defined by
                center x_T and radius r_D

    """
    pass
