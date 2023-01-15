import numpy as np

def normalise2DPts(pts):
    """  normalises 2D homogeneous points

     Function translates and normalises a set of 2D homogeneous points
     so that their centroid is at the origin and their mean distance from
     the origin is sqrt(2).

     Usage:   [pts_tilde, T] = normalise2dpts(pts)

     Argument:
       pts -  3xN array of 2D homogeneous coordinates

     Returns:
       pts_tilde -  3xN array of transformed 2D homogeneous coordinates.
       T         -  The 3x3 transformation matrix, pts_tilde = T*pts
    """

    pass
