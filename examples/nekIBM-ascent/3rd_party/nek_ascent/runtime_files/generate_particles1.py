import argparse
import numpy as np
import open3d as o3d


def main():
    parser = argparse.ArgumentParser(description="Generate a particles1.dat file")
    parser.add_argument(
        "voxel_size",
        type=float,
        help="The voxel size to use when downsampling",
    )
    args = parser.parse_args()

    data = np.loadtxt("particles-base.dat")
    result = data[data[:, 1] >= -3.2]

    # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(result)
    o3d.io.write_point_cloud("fragment.ply", pcd)

    # Load saved point cloud and visualize it
    pcd_load = o3d.io.read_point_cloud("fragment.ply")

    # Load a ply point cloud
    pcd = o3d.io.read_point_cloud("fragment.ply")

    # Downsample the point cloud with voxel_size
    downpcd = pcd.voxel_down_sample(voxel_size=args.voxel_size)

    # Recompute the normal of the downsampled point cloud
    downpcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )

    # saved downscaled geometry
    o3d.io.write_point_cloud("fragment_downsample.ply", downpcd)

    # Load downscaled geometry
    pcd_load = o3d.io.read_point_cloud("fragment_downsample.ply")

    xyz_downsampled = np.asarray(pcd_load.points)

    # Calculate the means for x and y dimensions
    centroid_x, centroid_y = np.mean(xyz_downsampled[:, 0]), np.mean(
        xyz_downsampled[:, 1]
    )

    # Center the x and y coordinates around [0,0,0]
    xyz_downsampled[:, 0] -= centroid_x
    xyz_downsampled[:, 1] -= centroid_y

    # Adjust z coordinates to start from 0
    xyz_downsampled[:, 2] -= np.min(xyz_downsampled[:, 2])

    # The float number 'a' as input, representing the diameter in this context
    a = 0.8E-3  # Example value, adjust as needed or prompt the user

    # Calculating required statistics on the adjusted data
    num_points = xyz_downsampled.shape[0]
    xlow = np.min(xyz_downsampled[:, 0])
    xhigh = np.max(xyz_downsampled[:, 0])
    zlow = np.min(xyz_downsampled[:, 2])
    zhigh = np.max(xyz_downsampled[:, 2])

    # Writing to the file, ensuring all floats have exactly five decimal places and are in scientific notation
    with open("particles1.dat", "w") as file:
        # Writing the header with num_points and bounds for x and z
        file.write(f"{num_points:6d} {xlow/100.:.6E} {xhigh/100.:.6E} {zlow/100.:.6E} {zhigh/100.:.6E}\n")

        # Writing each particle's information, using the adjusted coordinates
        for i, (x, y, z) in enumerate(xyz_downsampled, start=1):
            file.write(f"{i:6d} {a:.6E} {x/100.:.6E} {y/100.:.6E} {z/100.:.6E}\n")


if __name__ == "__main__":
    main()
