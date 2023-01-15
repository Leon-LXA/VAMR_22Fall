import numpy as np


def disparityToPointCloud(disp_img, K, baseline, left_img):
    """
    points should be Nx3 and intensities N, where N is the amount of pixels which have a valid disparity.
    I.e., only return points and intensities for pixels of left_img which have a valid disparity estimate!
    The i-th intensity should correspond to the i-th point.
    """
    h, w = disp_img.shape
    # Convenient way to build per-pixel coordinates.
    X, Y = np.meshgrid(np.arange(1, w + 1), np.arange(1, h + 1))
    X, Y = X.reshape(h*w), Y.reshape(h*w)
    px_left = np.stack([X, Y, np.ones_like(X)], axis=-1).astype(np.float)

    # Corresponding pixels in right image = pixel coords in left img minus disparity.
    px_right = px_left.copy()
    px_right[:, 0] -= disp_img.reshape(h*w)
    
    # Filter out pixels that do not have a known disparity.
    invalid_disp = disp_img.reshape(h*w) > 0
    px_left = px_left[invalid_disp, :]
    px_right = px_right[invalid_disp, :]
    intensities = left_img[invalid_disp.reshape([h, w])]
    
    # Reproject pixels: Get bearing vectors of rays in camera frame.
    K_inv = np.linalg.inv(K)
    bv_left = np.matmul(K_inv, px_left[:, :, None]).squeeze(-1)
    bv_right = np.matmul(K_inv, px_right[:, :, None]).squeeze(-1)

    # Intersect rays according to formula in problem statement.
    b = np.asarray([baseline, 0, 0])

    A = np.stack([bv_left, -bv_right], axis=-1)
    A_pseudo_inv = np.linalg.inv(np.matmul(A.transpose([0, 2, 1]), A))
    lambda_variable = np.matmul(A_pseudo_inv, np.matmul(A.transpose([0, 2, 1]), b[:, None]))

    points = bv_left * lambda_variable[:, 0, :]


    return points, intensities
