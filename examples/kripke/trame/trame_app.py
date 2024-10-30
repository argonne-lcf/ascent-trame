import asyncio
import time
import numpy as np
import cv2
from multiprocessing import Process, Queue
from multiprocessing.managers import BaseManager
from trame.app import get_server, asynchronous
from trame.widgets import vuetify, rca, client
from trame.ui.vuetify import SinglePageLayout

class QueueManager(BaseManager):
    pass
    
def main():
    # create queues for Trame state and updates
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
    while data != '':
        print('waiting on data... ', end='')
        sim_data = queue_data.get()
        print(f'received!')
        
        state_queue.put(sim_data)
        updates = update_queue.get()

        queue_signal.put(updates)

def runTrameServer(state_queue, update_queue):
    # create Ascent View
    view = AscentView()

    # set up Trame application
    server = get_server(client_type="vue2")
    state = server.state
    ctrl = server.controller

    # register RCA view with Trame controller
    view_handler = None
    @ctrl.add("on_server_ready")
    def initRca(**kwargs):
        nonlocal view_handler
        view_handler = RcaViewAdapter(view, 'view')
        ctrl.rc_area_register(view_handler)
    
        asynchronous.create_task(checkForStateUpdates(state, state_queue, update_queue, view, view_handler))

    # callback for steering enabled change
    def uiStateEnableSteeringUpdate(enable_steering, **kwargs):
        if state.connected:
            state.allow_submit = enable_steering
        if not enable_steering:
            update_queue.put({})

    # callback for color map change
    def uiStateColorMapUpdate(color_map, **kwargs):
        view.setColormap(color_map.lower())
        if view_handler is not None:
            view_handler.pushFrame()

    #
    def clearBarriers():
        view.clearBarriers()
        if view_handler is not None:
            view_handler.pushFrame()

    # callback for clicking submit button
    def submitSteeringOptions():
        steering_data = {
            'flow_speed': state.flow_speed,
            'barriers': view.getBarriers()
        }
        update_queue.put(steering_data)

    # register callbacks
    state.change('enable_steering')(uiStateEnableSteeringUpdate)    
    state.change('color_map')(uiStateColorMapUpdate)

    # define webpage layout
    state.allow_submit = False
    state.vis_style = 'width: 800px; height: 600px; border: solid 2px #000000; box-sizing: content-box;'
    with SinglePageLayout(server) as layout:
        client.Style('#rca-view div div img { width: 100%; height: auto; }')
        layout.title.set_text('Ascent-Trame')
        with layout.toolbar:
            vuetify.VDivider(vertical=True, classes="mx-2")
            vuetify.VSwitch(
                label='Enable Steering',
                v_model=('enable_steering', True),
                hide_details=True,
                dense=True
            )
            vuetify.VSpacer()
            vuetify.VSlider(
                label='Flow speed',
                v_model=('flow_speed', 0.75),
                min=0.25,
                max=1.50,
                step=0.05,
                hide_details=True,
                dense=True
            )
            vuetify.VCol(
                '{{flow_speed.toFixed(2)}}'
            )
            vuetify.VSpacer()
            vuetify.VSelect(
                label='Color Map',
                v_model=('color_map', 'Divergent'),
                items=('[\'Divergent\', \'Turbo\', \'Inferno\']',),
                hide_details=True,
                dense=True
            )
            vuetify.VSpacer()
            vuetify.VBtn(
                'Clear Barriers',
                color='secondary',
                click=clearBarriers
            )
            vuetify.VSpacer()
            vuetify.VBtn(
                'Submit',
                color='primary',
                disabled=('!allow_submit',),
                click=submitSteeringOptions
            )
        with layout.content:
            with vuetify.VContainer(fluid=True, classes='pa-0 fill-height', style='justify-content: center; align-items: start;'):
                v = rca.RemoteControlledArea(name='view', display='image', id='rca-view', style=('vis_style',))

    # start Trame server
    server.start()

async def checkForStateUpdates(state, state_queue, update_queue, view, view_handler):
    while True:
        try:
            state_data = state_queue.get(block=False)
           
            state.connected = True
            if state.enable_steering:
                state.allow_submit = True

            h, w = state_data['vorticity'].shape
            img_w = 1000
            img_h = img_w * h // w

            view.updateScale(img_w / w)
            view.updateData(state_data)
            view_handler.pushFrame()

            state.update({'vis_style': f'width: {img_w}px; height: {img_h}px; border: solid 2px #000000; box-sizing: content-box;'})
            state.flush()

            if not state.enable_steering:
                update_queue.put({})
        except:
            pass
        await asyncio.sleep(0)

def runQueueManager(queue_data, queue_signal):
    # register queues with Queue Manager
    QueueManager.register('get_data_queue', callable=lambda:queue_data)
    QueueManager.register('get_signal_queue', callable=lambda:queue_signal)
    
    # create Queue Manager
    mgr = QueueManager(address=('127.0.0.1', 8000), authkey=b'ascent-trame')
    
    # start Queue Manager server
    server = mgr.get_server()
    server.serve_forever()

# Trame RCA View Adapter
class RcaViewAdapter:
    def __init__(self, view, name):
        self._view = view
        self._streamer = None
        self._metadata = {
            'type': 'image/jpeg',
            'codec': '',
            'w': 0,
            'h': 0,
            'st': 0,
            'key': 'key'
        }

        self.area_name = name

    def pushFrame(self):
        if self._streamer is not None:
            asynchronous.create_task(self._asyncPushFrame())

    async def _asyncPushFrame(self):
        frame_data = self._view.getFrame()
        self._streamer.push_content(self.area_name, self._getMetadata(), frame_data.data)
    
    def _getMetadata(self):
        width, height = self._view.getSize()
        self._metadata['w'] = width
        self._metadata['h'] = height
        self._metadata['st'] = self._view.getFrameTime()
        return self._metadata

    def set_streamer(self, stream_manager):
        self._streamer = stream_manager

    def update_size(self, origin, size):
        width = int(size.get('w', 400))
        height = int(size.get('h', 300))
        print(f'new size: {width}x{height}')

    def on_interaction(self, origin, event):
        event_type = event['type']
        rerender = False

        if event_type == 'LeftButtonPress':
            rerender = self._view.onLeftMouseButton(event['x'], event['y'], True)
        elif event_type == 'LeftButtonRelease':
            rerender = self._view.onLeftMouseButton(event['x'], event['y'], False)
        elif event_type == 'MouseMove':
            rerender = self._view.onMouseMove(event['x'], event['y'])

        if rerender:
            frame_data = self._view.getFrame()
            self._streamer.push_content(self.area_name, self._getMetadata(), frame_data.data)


# Trame Custom View
class AscentView:
    def __init__(self):
        self._data = None
        self._scale = 1.0
        self._base_image = None
        self._image = np.zeros((2,1), dtype=np.uint8)
        self._jpeg_quality = 94
        self._frame_time = round(time.time_ns() / 1000000)
        self._colormaps = {
            'divergent': self._loadColorMap('resrc/colormap_divergent.png'),
            'turbo': self._loadColorMap('resrc/colormap_turbo.png'),
            'inferno': self._loadColorMap('resrc/colormap_inferno.png')
        }
        self._cmap = 'divergent'
        self._new_barrier = {'display': False, 'p0': None, 'p1': None}
        self._mouse_down = False
        self._mouse_start = {'x': 0, 'y': 0}

    def _loadColorMap(self, filename):
        cmap = cv2.imread(filename, cv2.IMREAD_COLOR)
        return cmap.reshape((cmap.shape[1], 3))

    def _calculateBarrierEnd(self, start, end):
        dx = abs(end['x'] - start['x'])
        dy = abs(end['y'] - start['y'])
        pos = {'x': end['x'], 'y': end['y']}
        if dx >= dy:
            pos['y'] = start['y']
        else:
            pos['x'] = start['x']
        return pos

    def _renderBarriers(self):
        # draw lines for barriers
        self._image = self._base_image.copy()
        for barrier in self._data['barriers']:
            self._image = cv2.line(self._image, (barrier[0], barrier[1]), (barrier[2], barrier[3]), (0, 0, 0), 1)
        if self._new_barrier['display']:
            pt0 = (self._new_barrier['p0']['x'], self._new_barrier['p0']['y'])
            pt1 = (self._new_barrier['p1']['x'], self._new_barrier['p1']['y'])
            self._image = cv2.line(self._image, pt0, pt1, (0, 0, 0), 1)

    """
    return: size of view (width, height)
    """
    def getSize(self):
        height, width, channels = self._image.shape
        return (width, height)

    """
    return: jpeg encoded binary data
    """
    def getFrame(self):
        result, encoded_img = cv2.imencode('.jpg', self._image, (cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality))
        if result:
            return encoded_img
        return None

    """
    return: time frame was created
    """
    def getFrameTime(self):
        return self._frame_time

    """
    return list of barriers
    """
    def getBarriers(self):
        barriers = None
        if self._data is not None:
            barriers = self._data['barriers']
        else:
            barriers = np.empty(shape=(0,0), dtype=np.int32)
        return barriers

    """
    Update scale for size image is displayed vs. actual size of image
    """
    def updateScale(self, scale):
        self._scale = scale

    """
    Update data and create new visualization
    return: None
    """
    def updateData(self, data):
        self._data = data
        # apply colormap to data
        val_min = -0.22
        val_max = 0.22
        vorticity = np.clip(data['vorticity'], val_min, val_max)
        colormap = self._colormaps[self._cmap]
        size = colormap.shape[0]
        d_norm = ((size - 1) * ((vorticity - val_min) / (val_max - val_min))).astype(dtype=np.uint16)
        self._base_image = colormap[d_norm]
        # draw lines for barriers
        self._renderBarriers()

    """
    Set color map to one from a predefined set
    return: None
    """
    def setColormap(self, cmap_name):
        self._cmap = cmap_name
        if self._data is not None:
            self.updateData(self._data)

    """

    """
    def clearBarriers(self):
        if self._data is not None:
            self._data['barriers'] = np.empty(shape=(0,0), dtype=np.int32)
            self._renderBarriers()

    """
    Handler for left mouse button
    return: whether or not rerender is required
    """
    def onLeftMouseButton(self, mouse_x, mouse_y, pressed):
        height = self._image.shape[0]
        mx = int(mouse_x / self._scale)
        my = height - int(mouse_y / self._scale)
        rerender = False
        if pressed:
            self._mouse_start['x'] = mx
            self._mouse_start['y'] = my
            self._new_barrier['display'] = True
            self._new_barrier['p0'] = self._mouse_start
            self._new_barrier['p1'] = self._mouse_start
        elif self._mouse_down:
            b_end = self._calculateBarrierEnd(self._mouse_start, {'x': mx, 'y': my})
            if self._data is not None:
                n_barrier = np.array([[self._mouse_start['x'], self._mouse_start['y'], b_end['x'], b_end['y']]], dtype=np.int32)
                if self._data['barriers'].size == 0:
                    self._data['barriers'] = n_barrier
                else:
                    self._data['barriers'] = np.concatenate((self._data['barriers'], n_barrier))
            self._new_barrier['display'] = False
            self._renderBarriers()
            rerender = True
        self._mouse_down = pressed
        return rerender

    """
    Handler for mouse movement
    return: whether or not rerender is required
    """
    def onMouseMove(self, mouse_x, mouse_y):
        height = self._image.shape[0]
        mx = int(mouse_x / self._scale)
        my = height - int(mouse_y / self._scale)
        rerender = False
        if self._mouse_down:
            b_end = self._calculateBarrierEnd(self._mouse_start, {'x': mx, 'y': my})
            self._new_barrier['p1'] = b_end
            self._renderBarriers()
            rerender = True
        return rerender


if __name__ == '__main__':
    main()
