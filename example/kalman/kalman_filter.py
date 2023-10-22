import numpy as np


def kalman_filter(x, P, F, H, R, Q, measurement, dt):
    # Update state transition matrix based on dynamic time step
    F[0, 1] = dt
    F[1, 2] = dt
    F[3, 4] = dt
    F[4, 5] = dt
    F[6, 7] = dt

    # Prediction step
    x_predicted = np.dot(F, x)
    P_predicted = np.dot(np.dot(F, P), F.T) + Q

    # Update step
    y = measurement - np.dot(H, x_predicted)
    S = np.dot(np.dot(H, P_predicted), H.T) + R
    K = np.dot(np.dot(P_predicted, H.T), np.linalg.inv(S))
    x = x_predicted + np.dot(K, y)
    P = np.dot((np.eye(x.shape[0]) - np.dot(K, H)), P_predicted)
    return x, P


# Initial error covariance matrix
P = np.eye(8)

# Define the state transition matrix
F = np.array(
    [
        [1, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 1],
    ],
    dtype=float,
)

# Define the measurement matrix (speed, acceleration, yaw, yaw rate)
H = np.array(
    [
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
    ]
)

# Define measurement noise covariance
R = np.eye(6) * 0.1

# Define process noise covariance
Q = np.eye(8) * 0.01

kalman_state = np.array(
    [
        0,
        12,
        0.1,
        0,
        0.2,
        0.1,
        3.14,
        0.12,
    ]
).T  # Initial x, vx, ax, y, vy, ay, yaw, yaw rate

measurement = np.array(
    [
        12.0,
        0.2,
        0.1,
        0.1,
        3.141,
        0.11,
        0.15,
    ]
).T

kalman_state, P = kalman_filter(
    kalman_state, P, F, H, R, Q, measurement[:-1], measurement[-1]
)

print(kalman_state)
