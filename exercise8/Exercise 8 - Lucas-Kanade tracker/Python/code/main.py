import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

from get_sim_warp import getSimWarp
from warp_image import warpImage
from get_warped_patch import getWarpedPatch
from track_brute_force import trackBruteForce
from track_klt import trackKLT
from track_klt_robustly import trackKLTRobustly

def part1():
    """ Warping Images """
    I_R = cv2.imread('../data/000000.png', cv2.IMREAD_GRAYSCALE)

    fig = plt.figure()
    ax = fig.add_subplot('221')
    ax.imshow(I_R, cmap='gray', vmin=0, vmax=255)
    ax.set_title("Reference Image")
    
    ax = fig.add_subplot('222')
    W = getSimWarp(50, -30, 0, 1)
    ax.imshow(warpImage(I_R,W), cmap='gray', vmin=0, vmax=255)
    ax.set_title("Translation")

    ax = fig.add_subplot('223')
    W = getSimWarp(0, 0, 10, 1)
    ax.imshow(warpImage(I_R,W), cmap='gray', vmin=0, vmax=255)
    ax.set_title("Rotation around upper left corner")

    ax = fig.add_subplot('224')
    W = getSimWarp(0, 0, 0, 0.5)
    ax.imshow(warpImage(I_R,W), cmap='gray', vmin=0, vmax=255)
    ax.set_title("Zoom on upper left corner")


def part2():
    """ Warping Images and recover warp with brute force matching"""
    I_R = cv2.imread('../data/000000.png', cv2.IMREAD_GRAYSCALE)

    fig = plt.figure()
    ax = fig.add_subplot('121')
    W0 = getSimWarp(0, 0, 0, 1)
    x_T = np.array([900, 291])
    r_T = 15
    template = getWarpedPatch(I_R, W0, x_T, r_T)
    ax.imshow(template)
    ax.set_title("Template")

    ax = fig.add_subplot('122')
    W = getSimWarp(10, 6, 0, 1)
    I = warpImage(I_R, W)
    r_D = 20
    t0 = time.time()
    dx, ssds = trackBruteForce(I_R, I, x_T, r_T, r_D)
    t1 = time.time()
    ax.imshow(ssds)
    ax.set_title("SSD's")
    print("Time taken: %5.2f [s]" % (t1 - t0))
    print("Displacement best plained by (dx, dy) = ( %5.2f, %5.2f )" % (dx[0], dx[1]))


def part3():
    """ Recovering the warp with KLT """
    I_R = cv2.imread('../data/000000.png', cv2.IMREAD_GRAYSCALE)
    x_T = np.array([899, 290])
    r_T = 15
    n_iter = 50

    W = getSimWarp(10, 6, 0, 1)
    I = warpImage(I_R, W)
    t0 = time.time()
    W, p_hist = trackKLT(I_R, I, x_T, r_T, n_iter) 
    t1 = time.time()
    print("Time taken: %5.2f [s]" % (t1 - t0))
    print("Displacement best plained by (dx, dy) = ( %5.2f, %5.2f )" % (W[0, -1], W[1, -1]))
    print("Displacement should be [-10, 6]")


def part4():
    I_R = cv2.imread('../data/000000.png', cv2.IMREAD_GRAYSCALE)
    I_R = cv2.resize(I_R, (0,0), fx = 0.25, fy = 0.25)
    keypoints = np.loadtxt('../data/keypoints.txt') / 4
    keypoints = np.flipud(keypoints[:50,:].T)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(I_R, cmap='gray', vmin=0, vmax=255 )
    ax.plot(keypoints[0, :], keypoints[1, :], 'rx')
    I_prev = I_R
    plt.pause(0.1)

    r_T = 15
    n_iter = 50
    ax.lines = []
    for i in range(20):
        I = cv2.imread("../data/%06d.png" % i, cv2.IMREAD_GRAYSCALE)
        I = cv2.resize(I, (0,0), fx = 0.25, fy = 0.25)
        dkp = np.zeros_like(keypoints)
        for j in range(keypoints.shape[1]):
            W, _ = trackKLT(I_prev, I, keypoints[:,j].T, r_T, n_iter)
            dkp[:, j] = W[:, -1];
        kpold = keypoints
        keypoints = keypoints + dkp
        ax.imshow(I, cmap='gray', vmin=0, vmax=255)
        keypoints_ud = np.flipud(keypoints)
        kpold_ud = np.flipud(kpold)
        x_from = keypoints_ud[0, :]
        x_to = kpold_ud[0,:]
        y_from = keypoints_ud[1, :]
        y_to = kpold_ud[1,:]
        ax.lines = []
        ax.plot(np.r_[y_from[np.newaxis, :], y_to[np.newaxis,:]], 
                np.r_[x_from[np.newaxis,:], x_to[np.newaxis,:]], 'g-',
                linewidth=3)
        ax.set_xlim([0, I.shape[1]])
        ax.set_ylim([I.shape[0], 0])
        I_prev = I
        plt.pause(0.01)


def part5():
    I_R = cv2.imread('../data/000000.png', cv2.IMREAD_GRAYSCALE)
    I_R = cv2.resize(I_R, (0,0), fx = 0.25, fy = 0.25)
    keypoints = np.loadtxt('../data/keypoints.txt') / 4
    keypoints = np.flipud(keypoints[:50,:].T)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(I_R, cmap='gray', vmin=0, vmax=255 )
    ax.plot(keypoints[0, :], keypoints[1, :], 'rx')
    I_prev = I_R
    plt.pause(0.1)

    r_T = 15
    n_iter = 50
    threshold = 0.1
    ax.lines = []
    for i in range(20):
        I = cv2.imread("../data/%06d.png" % i, cv2.IMREAD_GRAYSCALE)
        I = cv2.resize(I, (0,0), fx = 0.25, fy = 0.25)
        dkp = np.zeros_like(keypoints)
        keep = np.ones((keypoints.shape[1],)).astype('bool')
        for j in range(keypoints.shape[1]):
            kptd, k = trackKLTRobustly(I_prev, I, keypoints[:,j].T, r_T, n_iter, threshold)
            dkp[:, j] = kptd
            keep[j] = k
        kpold = keypoints[:, keep]
        keypoints = keypoints + dkp
        keypoints = keypoints[:, keep]

        ax.imshow(I, cmap='gray', vmin=0, vmax=255)
        keypoints_ud = np.flipud(keypoints)
        kpold_ud = np.flipud(kpold)
        x_from = keypoints_ud[0, :]
        x_to = kpold_ud[0,:]
        y_from = keypoints_ud[1, :]
        y_to = kpold_ud[1,:]
        ax.lines = []
        ax.plot(np.r_[y_from[np.newaxis, :], y_to[np.newaxis,:]], 
                np.r_[x_from[np.newaxis,:], x_to[np.newaxis,:]], 'g-',
                linewidth=3)
        ax.set_xlim([0, I.shape[1]])
        ax.set_ylim([I.shape[0], 0])
        I_prev = I
        plt.pause(0.01)
    
if __name__=="__main__":
    #  part1()
    #  part2()
    #  part3()
    #  part4()
    #  part5()
    
    #  plt.show()
