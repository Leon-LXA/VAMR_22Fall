import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


def getDisparity(left_img, right_img, patch_radius, min_disp, max_disp):
    """
    left_img and right_img are both H x W and you should return a H x W matrix containing the disparity d for
    each pixel of left_img. Set disp_img to 0 for pixels where the SSD and/or d is not defined, and for
    d estimates rejected in Part 2. patch_radius specifies the SSD patch and each valid d should satisfy
    min_disp <= d <= max_disp.
    """
    r = patch_radius
    patch_size = 2 * patch_radius + 1

    disp_img = np.zeros_like(left_img).astype(np.float)
    rows, cols = left_img.shape

    debug_ssds = False
    reject_outliers = True
    refine_estimate = True

    # pool = multiprocessing.Pool(processes=4)
    # pool.map(func, range(10))
    # pool.close()
    # pool.join()

    for row in range(patch_radius, rows - patch_radius):
        for col in range(max_disp + patch_radius, cols - patch_radius):
            # Here we construct what you can see in the left two subplots of Fig. 4.
            left_patch = left_img[(row - r):(row + r + 1), (col - r):(col + r + 1)]
            right_strip = right_img[(row - r):(row + r + 1), (col - r - max_disp):(col + r - min_disp + 1)]

            rsvecs = np.zeros([patch_size, patch_size, max_disp - min_disp + 1])
            for i in range(0, patch_size):
                rsvecs[:, i, :] = right_strip[:, i:(max_disp - min_disp + i + 1)]

            # Transforming the patches into vectors so we can run them through pdist2.
            lpvec = left_patch.flatten()
            rsvecs = rsvecs.reshape([patch_size**2, max_disp - min_disp + 1])

            ssds = cdist(lpvec[None, :], rsvecs.T, 'sqeuclidean').squeeze(0)

            if debug_ssds:
                plt.figure(figsize=(15, 4))
                plot1 = plt.subplot2grid((1, 4), (0, 0))
                plot2 = plt.subplot2grid((1, 4), (0, 1), colspan=2)
                plot3 = plt.subplot2grid((1, 4), (0, 3))

                plot1.imshow(left_patch)
                plot1.axis('off')
                plot2.imshow(right_strip)
                plot2.axis('off')

                plot3.plot(ssds)
                plt.xlabel('d-d_{max}')
                plt.ylabel('Shi-Tomasi Scores')

                plt.tight_layout()
                plt.show()

            # The way the patches are set up, the argmin of ssds will not directly be the disparity,
            # but rather (max_disparity - disparity). We call this "neg_disp".
            neg_disp = np.argmin(ssds)
            min_ssd = ssds[neg_disp]

            if reject_outliers:
                if (ssds <= 1.5 * min_ssd).sum() < 3 and neg_disp != 0 and neg_disp != ssds.shape[0] - 1:
                    if not refine_estimate:
                        disp_img[row, col] = max_disp - neg_disp - 1
                    else:
                        x = np.asarray([neg_disp - 1, neg_disp, neg_disp + 1])
                        p = np.polyfit(x, ssds[x], 2)

                        # Minimum of p(0)x^2 + p(1)x + p(2), converted from neg_disp to disparity as above.
                        disp_img[row, col] = max_disp + p[1] / (2 * p[0]) - 1

            else:
                disp_img[row, col] = max_disp - neg_disp - 1

    return disp_img
