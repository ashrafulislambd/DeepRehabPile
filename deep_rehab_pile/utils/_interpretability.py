"""Interpretability fidelity utilities (deletion test)."""

__all__ = ["mask_top_k_joints", "deletion_curve"]

import numpy as np


def mask_top_k_joints(
    X: np.ndarray, joint_ranking: np.ndarray, n_dim: int, k: int
) -> np.ndarray:
    """
    Zero out the top-k most important joints.

    Parameters
    ----------
    X: np.ndarray, shape = (n_samples, n_channels, n_timepoints),
        The input samples, channel-major order (joint, dim) as used
        throughout the codebase.
    joint_ranking: np.ndarray, shape = (n_joints,),
        Joint indices ordered from most to least important.
    n_dim: int,
        The number of dimensions per joint.
    k: int,
        The number of top joints to mask out.

    Returns
    -------
    X_masked: np.ndarray, shape = (n_samples, n_channels, n_timepoints),
        A copy of X with the top-k joints' channels zeroed out.
    """
    X_masked = X.copy()

    if k <= 0:
        return X_masked

    joints_to_mask = joint_ranking[:k]
    for joint in joints_to_mask:
        channels = np.arange(joint * n_dim, (joint + 1) * n_dim)
        X_masked[:, channels, :] = 0.0

    return X_masked


def deletion_curve(
    classifier,
    X: np.ndarray,
    y: np.ndarray,
    joint_ranking: np.ndarray,
    n_dim: int,
) -> np.ndarray:
    """
    Compute the accuracy deletion curve for a given joint ranking.

    Progressively masks out the top-k most important joints (per
    joint_ranking) and records how accuracy degrades, from k=0 (no
    masking) to k=n_joints (all joints masked).

    Parameters
    ----------
    classifier: BASE_CLASSIFIER,
        A fitted classifier exposing a ``score`` method.
    X: np.ndarray, shape = (n_samples, n_channels, n_timepoints),
        The input samples.
    y: np.ndarray, shape = (n_samples,),
        The true labels.
    joint_ranking: np.ndarray, shape = (n_joints,),
        Joint indices ordered from most to least important.
    n_dim: int,
        The number of dimensions per joint.

    Returns
    -------
    accuracies: np.ndarray, shape = (n_joints + 1,),
        Accuracy after masking the top-k joints, for k = 0..n_joints.
    """
    n_joints = len(joint_ranking)
    accuracies = np.zeros(shape=(n_joints + 1,))

    for k in range(n_joints + 1):
        X_masked = mask_top_k_joints(X, joint_ranking, n_dim=n_dim, k=k)
        accuracies[k] = classifier.score(X_masked, y, metric="accuracy")

    return accuracies
