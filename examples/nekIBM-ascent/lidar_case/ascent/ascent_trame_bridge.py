from hmac import new
import os
import sys
import zipfile

from multiprocessing.managers import BaseManager

import ascent.mpi
import conduit
import numpy as np
import yaml

from mpi4py import MPI

import open3d as o3d

class QueueManager(BaseManager):
    pass

def adjust_voxel_size(voxel_size):
    data = np.loadtxt("particles/particles-base.dat")
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
    downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)

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
    a = 0.8E-3

    # Calculating required statistics on the adjusted data
    num_points = xyz_downsampled.shape[0]
    xlow = np.min(xyz_downsampled[:, 0])
    xhigh = np.max(xyz_downsampled[:, 0])
    zlow = np.min(xyz_downsampled[:, 2])
    zhigh = np.max(xyz_downsampled[:, 2])

    # Writing to the file, ensuring all floats have exactly five decimal places and are in scientific notation
    with open("particles1.dat", "w") as file:
        file.write(f"{num_points:6d} {xlow/100.:.6E} {xhigh/100.:.6E} {zlow/100.:.6E} {zhigh/100.:.6E}\n")

        for i, (x, y, z) in enumerate(xyz_downsampled, start=1):
            file.write(f"{i:6d} {a:.6E} {x/100.:.6E} {y/100.:.6E} {z/100.:.6E}\n")

def adjust_offset(offset_x=0.0, offset_y=0.0):
    print(f"[adjust_offset] Starting with offset_x = {offset_x}, offset_y = {offset_y}")
    filename = "particles1.dat"
    
    with open(filename, 'r') as file:
        lines = file.readlines()
    print(f"[adjust_offset] Read {len(lines)} lines from {filename}")
    
    # Check header line
    if not lines:
        print("[adjust_offset] Error: The file is empty.")
        return

    header_parts = lines[0].strip().split()
    if len(header_parts) < 1:
        print("[adjust_offset] Error: Header line is empty or malformed.")
        return
    
    num_particles = int(header_parts[0])
    print(f"[adjust_offset] Number of particles (from header): {num_particles}")

    with open(filename, 'w') as file:
        file.write(lines[0])
        for line_num, line in enumerate(lines[1:], start=2):
            if not line.strip():
                print(f"[adjust_offset] Encountered empty line at line {line_num}. Writing it unchanged.")
                file.write(line)
                continue

            parts = line.split()
            if len(parts) == 5:
                index, param, x, y, z = parts
                
                try:
                    new_x = float(x) + offset_x
                    new_y = float(y) + offset_y
                except ValueError as ve:
                    print(f"[adjust_offset] Error converting values on line {line_num}: {ve}")
                    file.write(line)
                    continue

                # Determine the original field widths for x and y
                x_field_width = len(x)
                y_field_width = len(y)
                
                x_formatted = f"{new_x:{x_field_width}.6E}"
                y_formatted = f"{new_y:{y_field_width}.6E}"
                
                try:
                    x_start = line.index(x)
                    x_end = x_start + len(x)
                    y_start = line.index(y, x_end)
                    y_end = y_start + len(y)
                except ValueError as ve:
                    print(f"[adjust_offset] Error finding substring on line {line_num}: {ve}")
                    file.write(line)
                    continue

                modified_line = (
                    line[:x_start] + x_formatted +
                    line[x_end:y_start] + y_formatted +
                    line[y_end:]
                )
                file.write(modified_line)
            else:
                print(f"[adjust_offset] Warning: Line {line_num} does not have exactly 5 fields. Writing line unchanged:\n{line.strip()}")
                file.write(line)
    print(f"[adjust_offset] Finished updating offsets in {filename}")


def main():
    # Obtain a mpi4py MPI comm object
    comm = MPI.Comm.f2py(ascent_mpi_comm_id())

    # Get task id and number of total tasks
    task_id = comm.Get_rank()

    # Run Trame tasks
    interactive = np.array([False], bool)
    update_data = None
    if task_id == 0:
        update_data = executeMainTask(comm)
    else:
        executeDependentTask(comm)

    # Broadcast updates to all ranks
    update_data = comm.bcast(update_data, root=0)

    # All ranks process the update_data and execute callbacks
    if update_data is not None:
        print(f"Rank {task_id} received update data: {update_data}")
        update_node = conduit.Node()
        output_node = conduit.Node()
        reload = False

        if "voxel_size" in update_data:
            print("Updating voxel size")
            voxel_size = float(update_data["voxel_size"])
            adjust_voxel_size(voxel_size)
            reload = True

        if "offset_x" in update_data or "offset_y" in update_data:
            print("Updating offsets")
            offset_x = 0.0
            offset_y = 0.0
            reload = True

            if "offset_x" in update_data:
                offset_x = float(update_data["offset_x"])
            if "offset_y" in update_data:
                offset_y = float(update_data["offset_y"])

            adjust_offset(offset_x, offset_y)

        ascent.mpi.execute_callback("load_new_data", update_node, output_node)


def executeMainTask(comm):
    interactive = np.array([False], bool)
    update_data = {}

    # Attempt to connect to Trame queue manager
    QueueManager.register("get_data_queue")
    QueueManager.register("get_signal_queue")
    mgr = QueueManager(address=("127.0.0.1", 8000), authkey=b"ascent-trame")
    try:
        mgr.connect()
        interactive[0] = True
    except Exception as e:
        print(f"Failed to connect to Trame queue manager: {e}")
        mgr = None

    # Broadcast to all processes whether Trame is currently running
    comm.Bcast((interactive, 1, MPI.BOOL), root=0)

    if interactive[0]:
        queue_data = mgr.get_data_queue()
        queue_signal = mgr.get_signal_queue()

        # Signal Trame that new cinema data is available
        queue_data.put({"new_timestep": True})
        update_data = queue_signal.get()

    print("Main task completed.")
    return update_data


def executeDependentTask(comm):
    interactive = np.array([False], bool)
    # Receive whether session is interactive or not from main task
    comm.Bcast((interactive, 1, MPI.BOOL), root=0)


if __name__ == "__main__":
    main()
