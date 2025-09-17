# Copied from maveric/artifact/mro/mobility.py

import logging
from typing import Tuple, Optional

import numpy as np


def U(rng, MIN, MAX, SAMPLES):
    return rng.random(SAMPLES.shape) * (MAX - MIN) + MIN


def gauss_markov(
    rng: np.random.Generator,
    num_users: int,
    dimensions: Tuple[int, int],
    velocity_mean: np.ndarray,
    alpha: float = 1.0,
    variance: float = 1.0,
    anchor_loc: Optional[np.ndarray] = None,
    cov_around_anchor: Optional[np.ndarray] = None,
):
    MAX_X, MAX_Y = dimensions
    USERS = np.arange(num_users)

    if anchor_loc is None:
        x = U(rng, 0, MAX_X, USERS)
        y = U(rng, 0, MAX_Y, USERS)
    else:
        old_num_users = num_users
        num_users = int(num_users / len(anchor_loc)) * len(anchor_loc)
        if old_num_users != num_users:
            logging.info("len(anchor_loc) must evenly divide num_users.....terminating....")
            return

        num_users_per_anchor = int(num_users / len(anchor_loc))
        if cov_around_anchor is None:
            cov_around_anchor = np.array([[1, 0], [0, 1]])
        x, y = non_homogeneous_drop(
            rng=rng,
            anchor_loc=anchor_loc,
            num_users_per_anchor=num_users_per_anchor,
            cov_around_anchor=cov_around_anchor,
        ).transpose()

    velocity = np.zeros(num_users) + velocity_mean
    theta = U(rng, 0, 2 * np.pi, USERS)
    angle_mean = theta

    alpha2 = 1.0 - alpha
    alpha3 = np.sqrt(1.0 - alpha * alpha) * variance

    while True:
        x = x + velocity * np.cos(theta)
        y = y + velocity * np.sin(theta)

        b = np.where(x < 0)[0]
        x[b] = -x[b]
        theta[b] = np.pi - theta[b]
        angle_mean[b] = np.pi - angle_mean[b]
        b = np.where(x > MAX_X)[0]
        x[b] = 2 * MAX_X - x[b]
        theta[b] = np.pi - theta[b]
        angle_mean[b] = np.pi - angle_mean[b]
        b = np.where(y < 0)[0]
        y[b] = -y[b]
        theta[b] = -theta[b]
        angle_mean[b] = -angle_mean[b]
        b = np.where(y > MAX_Y)[0]
        y[b] = 2 * MAX_Y - y[b]
        theta[b] = -theta[b]
        angle_mean[b] = -angle_mean[b]

        velocity = alpha * velocity + alpha2 * velocity_mean + alpha3 * rng.normal(0.0, 1.0, num_users)
        theta = alpha * theta + alpha2 * angle_mean + alpha3 * rng.normal(0.0, 1.0, num_users)

        yield np.dstack((x, y))[0]


def non_homogeneous_drop(
    rng: np.random.Generator,
    anchor_loc: np.ndarray,
    num_users_per_anchor: int,
    cov_around_anchor: np.ndarray,
):
    num_anchors = anchor_loc.shape[0]
    user_loc = []
    for anchor_it in range(num_anchors):
        anchor_mean = anchor_loc[anchor_it, :]
        user_loc.append(
            rng.multivariate_normal(mean=anchor_mean, cov=cov_around_anchor, size=num_users_per_anchor)
        )
    user_loc = np.concatenate(user_loc, axis=0)
    return user_loc

