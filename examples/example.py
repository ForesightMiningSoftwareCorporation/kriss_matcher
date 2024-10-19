# /// script
# dependencies = [
#   "numpy==1.26.4",
#   "open3d",
#   "kriss_matcher"
# ]
# requires-python = ">=3.12"
# ///


# almost verbatim copy of https://github.com/MIT-SPARK/TEASER-plusplus/
# blob/master/examples/teaser_python_fpfh_icp/example.py
import copy
import pathlib
import urllib.request

import open3d as o3d
import numpy as np

import kriss_matcher

VOXEL_SIZE = 0.05

CURRENT_DIR = pathlib.Path(__file__).parent.resolve()


def download_files():
    data_dir = CURRENT_DIR / "data"
    data_dir.mkdir(exist_ok=True, parents=True)
    file_names = ("cloud_bin_0.ply", "cloud_bin_4.ply")
    for file_name in file_names:
        file_path = data_dir / file_name
        if file_path.exists():
            continue
        urllib.request.urlretrieve(
            f"https://github.com/MIT-SPARK/TEASER-plusplus/raw/refs/heads/master/examples/teaser_python_fpfh_icp/data/{file_name}",
            str(file_path),
        )


def run():
    # Load and visualize two point clouds from 3DMatch dataset
    A_pcd_raw = o3d.io.read_point_cloud(str(CURRENT_DIR / "data/cloud_bin_0.ply"))
    B_pcd_raw = o3d.io.read_point_cloud(str(CURRENT_DIR / "data/cloud_bin_4.ply"))
    A_pcd_raw.paint_uniform_color([0.0, 0.0, 1.0])  # show A_pcd in blue
    B_pcd_raw.paint_uniform_color([1.0, 0.0, 0.0])  # show B_pcd in red
    o3d.visualization.draw_geometries([A_pcd_raw, B_pcd_raw])  # plot A and B

    # voxel downsample both clouds
    A_pcd = A_pcd_raw.voxel_down_sample(voxel_size=VOXEL_SIZE)
    B_pcd = B_pcd_raw.voxel_down_sample(voxel_size=VOXEL_SIZE)

    print("solve")
    R, t = kriss_matcher.find_point_cloud_transformation(
        np.asarray(A_pcd.points), np.asarray(B_pcd.points), VOXEL_SIZE
    )
    print("solved")
    T = np.identity(4)
    T[:3, :3] = R
    T[:3, 3] = t.ravel()
    print(T)
    A_pcd_T_teaser = copy.deepcopy(A_pcd).transform(T)
    o3d.visualization.draw_geometries([A_pcd_T_teaser, B_pcd])

    NOISE_BOUND = VOXEL_SIZE
    # local refinement using ICP
    icp_sol = o3d.pipelines.registration.registration_icp(
        A_pcd,
        B_pcd,
        NOISE_BOUND,
        T,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
    )
    T_icp = icp_sol.transformation

    # visualize the registration after ICP refinement
    A_pcd_T_icp = copy.deepcopy(A_pcd).transform(T_icp)
    o3d.visualization.draw_geometries([A_pcd_T_icp, B_pcd])


if __name__ == "__main__":
    download_files()
    run()
