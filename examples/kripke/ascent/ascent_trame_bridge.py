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


def update_cinema_yaml(
    file_path, new_phi=None, new_theta=None, new_zoom=None, new_annotations=None
):
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)

    for entry in data:
        if "scenes" in entry:
            for scene in entry["scenes"].values():
                if "renders" in scene:
                    for render in scene["renders"].values():
                        if new_phi is not None and "phi" in render:
                            render["phi"] = new_phi
                        if new_theta is not None and "theta" in render:
                            render["theta"] = new_theta
                        if new_zoom is not None and "camera" in render:
                            render["camera"]["zoom"] = new_zoom
                        if new_annotations is not None and "annotations" in render:
                            render["annotations"] = new_annotations

    with open(file_path, "w") as file:
        yaml.dump(data, file, default_flow_style=False)


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

        if any(key in update_data for key in ["phi", "theta", "zoom", "annotations"]):
            yaml_file = "cinema.yaml"
            print(f"Updating cinema yaml file: {yaml_file}")
            print(f"Updates: {update_data}")
            update_cinema_yaml(
                yaml_file,
                new_phi=update_data.get("phi"),
                new_theta=update_data.get("theta"),
                new_zoom=update_data.get("zoom"),
                new_annotations=update_data.get("annotations"),
            )

    return update_data


def executeDependentTask(comm):
    interactive = np.array([False], bool)

    # receive whether session is interactive or not from main task
    comm.Bcast((interactive, 1, MPI.BOOL), root=0)

    if interactive[0]:
        pass


if __name__ == "__main__":
    main()
