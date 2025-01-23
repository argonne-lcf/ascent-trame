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

sys.path.append(
    f"../../.venv/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
)


class QueueManager(BaseManager):
    pass


def main():
    # obtain a mpi4py mpi comm object
    comm = MPI.Comm.f2py(ascent_mpi_comm_id())

    # get task id and number of total tasks
    task_id = comm.Get_rank()

    # run Trame tasks
    interactive = np.array([False], bool)
    update_data = None
    if task_id == 0:
        update_data = executeMainTask(comm)
    else:
        executeDependentTask(comm)

    # broadcast updates to all ranks
    update_data = comm.bcast(update_data, root=0)


# Realized that the client is pretty much always going to run where the server is, making this not necessary
def create_zip_file(source_dir, zip_name="cinema_data.zip"):
    with zipfile.ZipFile(zip_name, "w") as zipf:
        for root, _, files in os.walk(source_dir):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, arcname=os.path.relpath(file_path, source_dir))
    return zip_name


def executeMainTask(comm):
    interactive = np.array([False], bool)
    update_data = {}

    # attempt to connect to Trame queue manager
    QueueManager.register("get_data_queue")
    QueueManager.register("get_signal_queue")
    mgr = QueueManager(address=("127.0.0.1", 8000), authkey=b"ascent-trame")
    try:
        mgr.connect()
        interactive[0] = True
    except Exception as e:
        print(f"Failed to connect to Trame queue manager: {e}")
        mgr = None

    # broadcast to all processes whether Trame is currently running
    comm.Bcast((interactive, 1, MPI.BOOL), root=0)

    if interactive[0]:
        queue_data = mgr.get_data_queue()
        queue_signal = mgr.get_signal_queue()

        zip_file = create_zip_file("cinema_databases")
        if not os.path.exists(zip_file):
            print(f"Zip file creation failed: {zip_file} not found.")
            return
        print(f"Zip file created at: {zip_file}")

        try:
            with open(zip_file, "rb") as f:
                zip_data = f.read()
                queue_data.put({"zip_content": zip_data})
                os.remove(zip_file)
            print("Zip file sent to server.")
        except Exception as e:
            print(f"Failed to send zip file: {e}")

        update_data = queue_signal.get()

        update_node = conduit.Node()
        output_node = conduit.Node()

        if "reduce_particles" in update_data:
            print("Reducing particles")
            update_node["voxel_size"] = update_data["reduce_particles"]
            ascent.mpi.execute_callback("reduce_particles", update_node, output_node)
            ascent.mpi.execute_callback("load_new_data", update_node, output_node)
            print("Reloaded new particle dataset")


    return update_data


def executeDependentTask(comm):
    interactive = np.array([False], bool)

    # receive whether session is interactive or not from main task
    comm.Bcast((interactive, 1, MPI.BOOL), root=0)

    if interactive[0]:
        pass


if __name__ == "__main__":
    main()
