import pandas as pd
import os
import zipfile
import numpy as np
import cv2
from multiprocessing import Process, Queue
from multiprocessing.managers import BaseManager
from trame.app import get_server, asynchronous
from trame.widgets import vuetify, rca
from trame.ui.vuetify import SinglePageLayout


class QueueManager(BaseManager):
    pass


def main():
    state_queue = Queue()
    update_queue = Queue()

    # start Trame app in new thread
    trame_thread = Process(target=runTrameServer, args=(state_queue, update_queue))
    trame_thread.daemon = True
    trame_thread.start()

    # create queues from Ascent data
    queue_data = Queue()
    queue_signal = Queue()

    # start Queue Manager in new thread
    queue_mgr_thread = Process(target=runQueueManager, args=(queue_data, queue_signal))
    queue_mgr_thread.daemon = True
    queue_mgr_thread.start()

    # wait for data coming from Ascent
    data = None
    while data != "":
        print("waiting on data... ", end="")
        sim_data = queue_data.get()
        print(f"received!")

        state_queue.put(sim_data)
        updates = update_queue.get()

        queue_signal.put(updates)


def runTrameServer(state_queue, update_queue):
    view = AscentView()
    server = get_server(client_type="vue2")
    state = server.state
    ctrl = server.controller
    cinema_data = None
    view_handler = None

    @ctrl.add("on_server_ready")
    def initRca(**kwargs):
        nonlocal view_handler
        view_handler = RcaViewAdapter(view, "view")
        ctrl.rc_area_register(view_handler)
        asynchronous.create_task(
            checkForStateUpdates(state, state_queue, update_queue, view, view_handler)
        )

    @state.change("timestep", "phi", "theta")
    def uiStateUpdateImage(**kwargs):
        if cinema_data is not None and view_handler is not None:
            selected_row = cinema_data[
                (cinema_data["time"] == state.timestep)
                & (cinema_data["phi"] == state.phi)
                & (cinema_data["theta"] == state.theta)
            ]
            if not selected_row.empty:
                image_file = selected_row.iloc[0]["FILE"]
                image_path = os.path.join("received", "ascent-trame", image_file)
                view.setImagePath(image_path)
                view_handler.pushFrame()
            else:
                print("No matching image found for the selected values.")

    state.vis_style = "width: 800px; height: 600px; border: solid 2px #000000; box-sizing: content-box;"
    with SinglePageLayout(server) as layout:
        layout.title.set_text("Cinema Database Viewer")

        with layout.toolbar:
            vuetify.VSelect(
                v_model=("timestep", 0),
                items=("timestep_values", []),
                label="Timestep",
                dense=True,
                hide_details=True,
            )
            vuetify.VSelect(
                v_model=("phi", 0),
                items=("phi_values", []),
                label="Phi",
                dense=True,
                hide_details=True,
            )
            vuetify.VSelect(
                v_model=("theta", 0),
                items=("theta_values", []),
                label="Theta",
                dense=True,
                hide_details=True,
            )

        with layout.content:
            with vuetify.VContainer(
                fluid=True,
                classes="pa-0 fill-height",
                style="justify-content: center; align-items: start;",
            ):
                v = rca.RemoteControlledArea(
                    name="view", display="image", id="rca-view", style=("vis_style",)
                )

    async def checkForStateUpdates(
        state, state_queue, update_queue, view, view_handler
    ):
        nonlocal cinema_data
        while True:
            try:
                data = state_queue.get(block=False)

                if "zip_content" in data:
                    print("Received zip_content in state_data, unzipping...")
                    zip_path = "received/cinema_data.zip"
                    os.makedirs("received", exist_ok=True)

                    # Write zip content to a file
                    with open(zip_path, "wb") as f:
                        f.write(data["zip_content"])

                    # Extract zip file contents
                    with zipfile.ZipFile(zip_path, "r") as zip_ref:
                        zip_ref.extractall("received/ascent-trame")
                    print("Extraction complete.")

                    csv_path = "received/ascent-trame/data.csv"
                    cinema_data = pd.read_csv(csv_path)

                    if cinema_data is not None:
                        # Populate dropdown values based on data.csv
                        state.timestep_values = [
                            {"text": str(v), "value": v}
                            for v in sorted(cinema_data["time"].unique())
                        ]
                        state.phi_values = [
                            {"text": str(v), "value": v}
                            for v in sorted(cinema_data["phi"].unique())
                        ]
                        state.theta_values = [
                            {"text": str(v), "value": v}
                            for v in sorted(cinema_data["theta"].unique())
                        ]

                        # Set initial dropdown selections
                        state.timestep = state.timestep_values[0]["value"]
                        state.phi = state.phi_values[0]["value"]
                        state.theta = state.theta_values[0]["value"]

                        state.flush()
                        uiStateUpdateImage()
            except:
                pass

    server.start()


def runQueueManager(queue_data, queue_signal):
    # register queues with Queue Manager
    QueueManager.register("get_data_queue", callable=lambda: queue_data)
    QueueManager.register("get_signal_queue", callable=lambda: queue_signal)

    # create Queue Manager
    mgr = QueueManager(address=("127.0.0.1", 8001), authkey=b"ascent-trame")

    # start Queue Manager server
    server = mgr.get_server()
    server.serve_forever()


class RcaViewAdapter:
    def __init__(self, view, name):
        self._view = view
        self._streamer = None
        self.area_name = name
        self._metadata = {
            "type": "image/jpeg",
            "codec": "",
            "w": 0,
            "h": 0,
            "key": "key",
        }

    def pushFrame(self):
        if self._streamer is not None:
            asynchronous.create_task(self._asyncPushFrame())

    async def _asyncPushFrame(self):
        frame_data = self._view.getFrame()
        self._streamer.push_content(
            self.area_name, self._getMetadata(), frame_data.data
        )

    def _getMetadata(self):
        width, height = self._view.getSize()
        self._metadata["w"] = width
        self._metadata["h"] = height
        return self._metadata

    def set_streamer(self, stream_manager):
        self._streamer = stream_manager

    def update_size(self, origin, size):
        width = int(size.get("w", 400))
        height = int(size.get("h", 300))
        print(f"new size: {width}x{height}")

    def on_interaction(self, origin, event):
        print(f"Interaction received: {event}")


class AscentView:
    def __init__(self):
        self._path = None
        self._scale = 1.0
        self._base_image = None
        self._image = np.zeros((2, 1), dtype=np.uint8)
        self._jpeg_quality = 94

    def getSize(self):
        height, width, channels = self._image.shape
        return (width, height)

    def setImagePath(self, image_path):
        print(f"Setting image: {image_path}")
        if os.path.exists(image_path):
            self._path = image_path
        else:
            print(f"Image path not found: {image_path}")

    def getFrame(self):
        if not os.path.exists(self._path):
            return None

        image = cv2.imread(self._path)
        result, encoded_img = cv2.imencode(
            ".jpg", image, (cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality)
        )
        if result:
            return encoded_img
        return None

    def updateScale(self, scale):
        self._scale = scale

    def updateData(self, data):
        self._data = data


if __name__ == "__main__":
    main()
