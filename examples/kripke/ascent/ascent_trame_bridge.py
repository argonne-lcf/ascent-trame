import os
import zipfile
from multiprocessing.managers import BaseManager
from mpi4py import MPI


class QueueManager(BaseManager):
    pass


def create_zip_file(source_dir, zip_name="cinema_data.zip"):
    with zipfile.ZipFile(zip_name, "w") as zipf:
        for root, _, files in os.walk(source_dir):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, arcname=os.path.relpath(file_path, source_dir))
    return zip_name


def main():
    comm = MPI.COMM_WORLD
    task_id = comm.Get_rank()

    if task_id == 0:
        QueueManager.register("get_data_queue")
        mgr = QueueManager(address=("127.0.0.1", 8001), authkey=b"ascent-trame")
        mgr.connect()
        queue_data = mgr.get_data_queue()

        # Zip the contents of "cinema_databases"
        zip_file = create_zip_file("cinema_databases")
        if not os.path.exists(zip_file):
            print(f"Zip file creation failed: {zip_file} not found.")
            return
        print(f"Zip file created at: {zip_file}")

        # Send the zipped cinema database file to the server
        try:
            with open(zip_file, "rb") as f:
                zip_data = f.read()
                queue_data.put({"zip_content": zip_data})
            print("Zip file sent to server.")
        except Exception as e:
            print(f"Failed to send zip file: {e}")


if __name__ == "__main__":
    main()
