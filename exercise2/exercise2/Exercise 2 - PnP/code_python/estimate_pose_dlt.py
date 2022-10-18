import numpy as np

def estimatePoseDLT(p, P, K):
    # Estimates the pose of a camera using a set of 2D-3D correspondences
    # and a given camera matrix.
    # 
    # p  [n x 2] array containing the undistorted coordinates of the 2D points
    # P  [n x 3] array containing the 3D point positions
    # K  [3 x 3] camera matrix
    #
    # Returns a [3 x 4] projection matrix of the form 
    #           M_tilde = [R_tilde | alpha * t] 
    # where R is a rotation matrix. M_tilde encodes the transformation 
    # that maps points from the world frame to the camera frame

    # YOUR CODE GOES HERE! 
    # THE COMMENTS ARE MEANT TO HELP YOU
    
    # Convert 2D to normalized coordinates

    # Build measurement matrix Q

    # Solve for Q.M_tilde = 0 subject to the constraint ||M_tilde||=1
    
    # Extract [R | t] with the correct scale

    # Find the closest orthogonal matrix to R
    # https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    # Normalization scheme using the Frobenius norm:
    # recover the unknown scale using the fact that R_tilde is a true rotation matrix

    # Build M_tilde with the corrected rotation and scale and return it

    pass
    
     
