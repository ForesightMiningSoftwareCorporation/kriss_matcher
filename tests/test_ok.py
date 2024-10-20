import kriss_matcher
import numpy as np

VOXEL_SIZE = 0.05
NOISE_STD_DEV = 0.01


def generate_noisy_transformed_cloud(
    source_cloud, rotation_matrix, translation_vector, noise_stddev
):
    transformed_cloud = (rotation_matrix @ source_cloud.T).T + translation_vector

    noise = np.random.normal(0, noise_stddev, transformed_cloud.shape)
    noisy_transformed_cloud = transformed_cloud + noise

    return noisy_transformed_cloud


def test_ok():
    np.random.seed(42)
    num_points = 1000
    A_pcd = np.random.rand(num_points, 3)

    rotation_angle = np.pi / 4
    rotation_matrix = np.array(
        [
            [np.cos(rotation_angle), -np.sin(rotation_angle), 0],
            [np.sin(rotation_angle), np.cos(rotation_angle), 0],
            [0, 0, 1],
        ]
    )
    translation_vector = np.array([42.0, 24.0, 0.1])

    B_pcd = generate_noisy_transformed_cloud(
        A_pcd, rotation_matrix, translation_vector, NOISE_STD_DEV
    )

    R, t = kriss_matcher.find_point_cloud_transformation(A_pcd, B_pcd, VOXEL_SIZE)

    T_estimated = np.identity(4)
    T_estimated[:3, :3] = R
    T_estimated[:3, 3] = t.ravel()

    T_true = np.identity(4)
    T_true[:3, :3] = rotation_matrix
    T_true[:3, 3] = translation_vector

    rotation_error = np.linalg.norm(T_true[:3, :3] - T_estimated[:3, :3])
    translation_error = np.linalg.norm(T_true[:3, 3] - T_estimated[:3, 3])

    assert translation_error < 0.3
    assert rotation_error < 0.4
