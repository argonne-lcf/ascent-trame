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


class QueueManager(BaseManager):
    pass


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
    if update_data is not None and ("reduce_particles" in update_data or "tree_offset" in update_data):
        update_node = conduit.Node()
        output_node = conduit.Node()

        if "reduce_particles" in update_data:
            update_node["voxel_size"] = float(update_data["reduce_particles"])
        else:
            update_node["voxel_size"] = 1.0

        if "tree_offset" in update_data:
            update_node["tree_offset"] = float(update_data["tree_offset"])
        else:
            update_node["tree_offset"] = 0.0

        # Execute the callbacks on all ranks
        ascent.mpi.execute_callback("reduce_particles", update_node, output_node)
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
