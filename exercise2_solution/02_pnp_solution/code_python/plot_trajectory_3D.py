import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from draw_camera import drawCamera

import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Use Agg backend for canvas
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def plotTrajectory3D(fps, filename, translations, quaternions, pts3d):
    # Draw the trajectory of the camera (3 colored axes, RGB).
    #
    # fps           framerate of the video
    # filename      filename of the video file to be created
    # transl(N, 3):  translations
    # quats(N, 4):   orientations given by quaternions [x, y, z, w]
    # pts3d(N, 3):   additional 3D points to plot
    #
    # translation and quaterionss refer to the tranformation T_W_C 
    # that maps points from the camera coordinate frame to the world 
    # frame, i.e. the transformation that expresses the camera position 
    # in the world frame.
    
    xmin = min(np.min(translations[:,0]), np.min(pts3d[:,0]))
    xmax = max(np.max(translations[:,0]), np.max(pts3d[:,0]))
    ymin = min(np.min(translations[:,1]), np.min(pts3d[:,1]))
    ymax = max(np.max(translations[:,1]), np.max(pts3d[:,1]))
    zmin = min(np.min(translations[:,2]), np.min(pts3d[:,2]))
    zmax = max(np.max(translations[:,2]), np.max(pts3d[:,2]))

    video = None
    
    fig = plt.figure()
    for i in range(translations.shape[0]):
        fig.clear()
        ax = fig.add_subplot(projection='3d')
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.set_zlim([zmin, zmax])
    
        ax.set_box_aspect((np.ptp(ax.get_xlim()),
                       np.ptp(ax.get_ylim()),
                       np.ptp(ax.get_zlim())))
        ax.scatter(pts3d[:,0], pts3d[:,1], pts3d[:,2])
        rotMat = Rotation.from_quat(quaternions[i, :]).as_matrix()
        drawCamera(ax, translations[i,:], rotMat, length_scale = 0.1, head_size = 10,
                set_ax_limits = False)

        canvas = FigureCanvas(fig)
        canvas.draw()

        mat = np.array(canvas.renderer._renderer)
        mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)

        if video is None:
            video_size = (mat.shape[1], mat.shape[0])
            video = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc('X','V','I','D'), 
                30, video_size)

        mat = cv2.resize(mat, video_size)
        cv2.imshow("test", mat)
        cv2.waitKey(30)
        video.write(mat)

    cv2.destroyAllWindows()
    video.release()

