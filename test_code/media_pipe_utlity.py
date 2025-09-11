import numpy as np

def normalize_pose_landmarks(landmarks, min_visibility=0.5):
    """
    Normalizes MediaPipe 3D pose landmarks for top-down (CCTV) compatibility.

    Args:
        landmarks (list of 33 MediaPipe landmarks): Pose landmarks
        min_visibility (float): Threshold to keep landmarks (skip low-vis)

    Returns:
        np.array of shape (132,) — normalized and flattened (x, y, z, v)
    """
    # Extract coordinates and visibility
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    visibility = np.array([lm.visibility for lm in landmarks])

    # Zero out low-visibility landmarks
    for i, vis in enumerate(visibility):
        if vis < min_visibility:
            coords[i] = [0.0, 0.0, 0.0]


    # Flatten and add visibility
    flat_coords = coords.flatten()
    return np.concatenate([flat_coords, visibility])
