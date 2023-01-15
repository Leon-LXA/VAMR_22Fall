import numpy as np
import matplotlib.pyplot as plt

from utils import twist2HomogMatrix


def plotMap(hidden_state, observations, axis_range, title):
    plt.clf()
    plt.close()

    num_frames = int(observations[0])
    T_W_frames = hidden_state[:num_frames*6].reshape([-1, 6]).T
    p_W_landmarks = hidden_state[num_frames*6:].reshape([-1, 3]).T

    p_W_frames = np.zeros([3, num_frames])
    for i in range(num_frames):
        T_W_frame = twist2HomogMatrix(T_W_frames[:, i])
        p_W_frames[:, i] = T_W_frame[:3, -1]

    plt.plot(p_W_landmarks[2, :], -p_W_landmarks[0, :], '.')
    plt.plot(p_W_frames[2, :], -p_W_frames[0, :], 'rx', linewidth=3)

    plt.axis('equal')
    plt.xlim(axis_range[:2])
    plt.ylim(axis_range[2:])
    plt.title(title)
    plt.show()
