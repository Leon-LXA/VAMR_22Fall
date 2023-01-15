import numpy as np


def cropProblem(hidden_state, observations, ground_truth, cropped_num_frames):
    # Determine which landmarks to keep; assuming landmark indices increase with frame indices.
    num_frames = int(observations[0])
    assert cropped_num_frames < num_frames

    observation_i = 2
    for i in range(cropped_num_frames):
        num_observations = int(observations[observation_i])
        if i == (cropped_num_frames - 1):
            cropped_num_landmarks = int(observations[observation_i+1+num_observations*2:observation_i+num_observations*3+1].max())
        observation_i = observation_i + num_observations * 3 + 1

    cropped_hidden_state = np.concatenate([hidden_state[:6*cropped_num_frames],
                                           hidden_state[6*num_frames:6*num_frames+3*cropped_num_landmarks]], axis=0)
    cropped_observations = np.concatenate([np.asarray([cropped_num_frames, cropped_num_landmarks]),
                                           observations[2:observation_i]], axis=0)
    cropped_ground_truth = ground_truth[:, :cropped_num_frames]

    return cropped_hidden_state, cropped_observations, cropped_ground_truth
