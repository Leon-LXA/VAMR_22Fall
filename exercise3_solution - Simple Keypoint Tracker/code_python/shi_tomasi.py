import numpy as np
from scipy import signal


def shi_tomasi(img, patch_size):
    sobel_para = np.array([-1, 0, 1])
    sobel_orth = np.array([1, 2, 1])

    Ix = signal.convolve2d(img, sobel_para[None, :], mode="valid")
    Ix = signal.convolve2d(Ix, sobel_orth[:, None], mode="valid").astype(float)

    Iy = signal.convolve2d(img, sobel_para[:, None], mode="valid")
    Iy = signal.convolve2d(Iy, sobel_orth[None, :], mode="valid").astype(float)

    Ixx = Ix ** 2
    Iyy = Iy ** 2
    Ixy = Ix*Iy

    patch = np.ones([patch_size, patch_size])
    pr = patch_size // 2
    sIxx = signal.convolve2d(Ixx, patch, mode="valid")
    sIyy = signal.convolve2d(Iyy, patch, mode="valid")
    sIxy = signal.convolve2d(Ixy, patch, mode="valid")

    trace = sIxx + sIyy
    determinant = sIxx * sIyy - sIxy**2

    # the eigen values of a matrix M=[a,b;c,d] are lambda1/2 = (Tr(A)/2 +- ((Tr(A)/2)^2-det(A))^.5
    # The smaller one is the one with the negative sign
    scores = trace/2 - ((trace/2)**2 - determinant)**0.5
    scores[scores < 0] = 0

    scores = np.pad(scores, [(pr+1, pr+1), (pr+1, pr+1)], mode='constant', constant_values=0)

    return scores

