import numpy as np

from utils import cross2Matrix

def linearTriangulation(p1, p2, M1, M2):
    """ Linear Triangulation
     Input:
      - p1 np.ndarray(3, N): homogeneous coordinates of points in image 1
      - p2 np.ndarray(3, N): homogeneous coordinates of points in image 2
      - M1 np.ndarray(3, 4): projection matrix corresponding to first image
      - M2 np.ndarray(3, 4): projection matrix corresponding to second image

     Output:
      - P np.ndarray(4, N): homogeneous coordinates of 3-D points
    """

    pass
