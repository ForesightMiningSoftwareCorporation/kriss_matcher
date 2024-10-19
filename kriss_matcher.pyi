from typing import Tuple
import numpy as np

def find_point_cloud_transformation(
    source_points: np.ndarray, target_points: np.ndarray, voxel_size: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds the transformation (rotation and translation) between two point clouds.

    Args:
        source_points: A 2D numpy array of shape (n, 3), where n is the number of points.
        target_points: A 2D numpy array of shape (n, 3), where n is the number of points.
        voxel_size: A float representing the size of the voxel grid used in the transformation.

    Returns:
        A tuple of two numpy 2D arrays:
            - The first array is the rotation matrix.
            - The second array is the translation matrix.
    """
    ...
