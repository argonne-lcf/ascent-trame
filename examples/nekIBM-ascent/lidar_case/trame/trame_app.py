import asyncio
import os
import time
import zipfile
import cv2
import pandas as pd
from multiprocessing import Process, Queue
from multiprocessing.managers import BaseManager
from trame.app import get_server, asynchronous
from trame.widgets import vuetify, rca, client
from trame.ui.vuetify import SinglePageLayout


class QueueManager(BaseManager):
    pass


def main():
    state_queue = Queue()
    update_queue = Queue()

    trame_thread = Process(target=runTrameServer, args=(state_queue, update_queue))
    trame_thread.daemon = True
    trame_thread.start()

    queue_data = Queue()
    queue_signal = Queue()

    queue_mgr_thread = Process(target=runQueueManager, args=(queue_data, queue_signal))
    queue_mgr_thread.daemon = True
    queue_mgr_thread.start()

    data = None
    while data != "":
        print("waiting on data... ", end="")
        sim_data = queue_data.get()
        print("received!")
        state_queue.put(sim_data)
        updates = update_queue.get()
        queue_signal.put(updates)


def runTrameServer(state_queue, update_queue):
    view = AscentView()

    server = get_server(client_type="vue2")
    state = server.state
    ctrl = server.controller

    # register RCA view with Trame controller
    view_handler = None

    @ctrl.add("on_server_ready")
    def initRca(**kwargs):
        nonlocal view_handler
        view_handler = RcaViewAdapter(view, "view")
        ctrl.rc_area_register(view_handler)

        asynchronous.create_task(
            checkForStateUpdates(state, state_queue, update_queue, view, view_handler)
        )

    # callback for steering enabled change
    def uiStateEnableSteeringUpdate(enable_steering, **kwargs):
        if state.connected:
            state.allow_submit = enable_steering
        if not enable_steering:
            update_queue.put({})

    def submitReloadData():
        updates["load_new_data"] = True
        update_queue.put(updates)

    # callback for clicking submit button
    def submitSteeringOptions():
        updates = {}

        if state.voxel_size_input:
            try:
                updates["reduce_particles"] = float(state.voxel_size_input)
            except ValueError:
                pass

        if state.tree_offset:
            try:
                updates["tree_offset"] = float(state.tree_offset)
            except ValueError:
                pass
        update_queue.put(updates)

    def loadFrame(**kwargs):
        print("Loading frame")
        if (
            state.selected_time is None
            or state.selected_phi is None
            or state.selected_theta is None
        ):
            return
        time = state.selected_time
        phi = state.selected_phi
        theta = state.selected_theta
        image_path = f"received/ascent-trame/{time}/{phi}_{theta}_ascent-trame.png"
        print(f"Checking for image: {image_path}")
        if os.path.exists(image_path):
            print(f"Loading frame: time={time}, phi={phi}, theta={theta}")
            view.updateData(image_path)
            if view_handler is not None:
                view_handler.pushFrame()
        else:
            print(f"Image not found: {image_path}")

    # register callbacks
    state.change("enable_steering")(uiStateEnableSteeringUpdate)
    state.change("selected_time")(loadFrame)
    state.change("selected_phi")(loadFrame)
    state.change("selected_theta")(loadFrame)

    state.timestep_values = []
    state.phi_values = []
    state.theta_values = []

    # define webpage layout
    state.allow_submit = False
    state.vis_style = "width: 1024px; height: 1024px; border: solid 2px #000000; box-sizing: content-box;"
    with SinglePageLayout(server) as layout:
        client.Style("#rca-view div div img { width: 100%; height: auto; }")
        layout.title.set_text("Ascent-Trame")
        with layout.toolbar:
            vuetify.VDivider(vertical=True, classes="mx-2")
            vuetify.VSwitch(
                label="Enable Steering",
                v_model=("enable_steering", True),
                hide_details=True,
                dense=True,
            )
            vuetify.VSpacer()
            vuetify.VSlider(
                label='Tree Position',
                v_model=('tree_offset', 0),
                min=-1,
                max=2,
                step=0.05,
                hide_details=True,
                dense=True
            )
            vuetify.VSpacer()
            vuetify.VTextField(
                label="Adjust Voxel Size",
                v_model=("voxel_size_input", ""),
                hide_details=True,
                dense=True,
                type="number",
            )
            vuetify.VSpacer()
            vuetify.VSelect(
                label="Timestep",
                v_model=("selected_time", ""),
                items=("timestep_values",),
                hide_details=True,
                dense=True,
            )
            vuetify.VSpacer()
            vuetify.VSelect(
                label="Phi",
                v_model=("selected_phi", "90.0"),
                items=("phi_values",),
                hide_details=True,
                dense=True,
            )
            vuetify.VSpacer()
            vuetify.VSelect(
                label="Theta",
                v_model=("selected_theta", "90.0"),
                items=("theta_values",),
                hide_details=True,
                dense=True,
            )
            vuetify.VSpacer()
            vuetify.VBtn(
                "Submit",
                color="primary",
                disabled=("!allow_submit",),
                click=submitSteeringOptions,
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

    # start Trame server
    server.start()


async def checkForStateUpdates(state, state_queue, update_queue, view, view_handler):
    while True:
        try:
            state_data = state_queue.get(block=False)

            state.connected = True
            print("Connected to Ascent")
            if state.enable_steering:
                print("Steering enabled")
                state.allow_submit = True

            if "zip_content" in state_data:
                receive_dir = "received"
                zip_path = os.path.join(receive_dir, "cinema_data.zip")
                print(f"Received zip file: {zip_path}")
                os.makedirs(receive_dir, exist_ok=True)
                with open(zip_path, "wb") as f:
                    f.write(state_data["zip_content"])
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(receive_dir)

                csv_path = os.path.join(receive_dir, "ascent-trame/data.csv")
                print(f"Loading CSV file: {csv_path}")
                if os.path.exists(csv_path):
                    cinema_data = pd.read_csv(csv_path)
                    print("Populating dropdowns")
                    state.update(
                        {
                            "timestep_values": [
                                {"text": str(v), "value": str(v)}
                                for v in sorted(cinema_data["time"].unique())
                            ],
                            "phi_values": [
                                {"text": str(v), "value": str(v)}
                                for v in sorted(cinema_data["phi"].unique())
                            ],
                            "theta_values": [
                                {"text": str(v), "value": str(v)}
                                for v in sorted(cinema_data["theta"].unique())
                            ],
                        }
                    )

                    if state.selected_time == "":
                        state.selected_time = state.timestep_values[0]["value"]

                    if state.selected_phi == "":
                        state.selected_phi = state.phi_values[0]["value"]

                    if state.selected_theta == "":
                        state.selected_theta = state.theta_values[0]["value"]

            state.flush()

            if not state.enable_steering:
                print("Steering disabled")
                update_queue.put({})

            print("State updated")
        except:
            pass
        await asyncio.sleep(0)


def runQueueManager(queue_data, queue_signal):
    # register queues with Queue Manager
    QueueManager.register("get_data_queue", callable=lambda: queue_data)
    QueueManager.register("get_signal_queue", callable=lambda: queue_signal)

    # create Queue Manager
    mgr = QueueManager(address=("127.0.0.1", 8000), authkey=b"ascent-trame")

    # start Queue Manager server
    server = mgr.get_server()
    server.serve_forever()


class RcaViewAdapter:
    def __init__(self, view, name):
        self._view = view
        self._streamer = None
        self._metadata = {
            "type": "image/jpeg",
            "codec": "",
            "w": 0,
            "h": 0,
            "st": 0,
            "key": "key",
        }
        self.area_name = name

    def set_streamer(self, stream_manager):
        self._streamer = stream_manager

    def pushFrame(self):
        if self._streamer is not None:
            asynchronous.create_task(self._asyncPushFrame())

    async def _asyncPushFrame(self):
        frame_data = self._view.getFrame()
        if frame_data is not None:
            self._streamer.push_content(
                self.area_name, self._getMetadata(), frame_data.data
            )

    def _getMetadata(self):
        width, height = self._view.getSize()
        return {
            "type": "image/jpeg",
            "codec": "",
            "w": width,
            "h": height,
            "st": self._view.getFrameTime(),
            "key": "key",
        }

    def update_size(self, origin, size):
        width = int(size.get("w", 400))
        height = int(size.get("h", 300))
        print(f"new size: {width}x{height}")

    def on_interaction(self, origin, event):
        pass


class AscentView:
    def __init__(self):
        self._image = None
        self._jpeg_quality = 94
        self._frame_time = round(time.time_ns() / 1000000)

    def updateData(self, image_path):
        self._image = cv2.imread(image_path)
        if self._image is not None:
            self._frame_time = round(time.time_ns() / 1000000)

    def getSize(self):
        if self._image is not None:
            height, width, _ = self._image.shape
            return width, height
        return 0, 0

    def getFrame(self):
        result, encoded_img = cv2.imencode(
            ".jpg", self._image, (cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality)
        )
        if result:
            return encoded_img
        return None

    def getFrameTime(self):
        return self._frame_time


if __name__ == "__main__":
    main()
