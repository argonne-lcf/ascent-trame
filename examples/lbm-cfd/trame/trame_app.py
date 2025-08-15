import asyncio
import time
import os
import numpy as np
import cv2
from ascenttrame.consumer import AscentConsumer
from ascenttrame.tramestreamer import TDivider, TSpacer, TText, TSwitch, TButton, TSlider, TDropDownMenu, TFileInput, TrameImageStreamer, TrameImageView


def main():
    # create bridge to consume simulation data from Ascent
    bridge = AscentConsumer(8000)
    
    # create custom View for TrameImageStreamer
    view = LbmCfdView()

    # set up Trame application
    trame_app = TrameImageStreamer(view, fixed_width=1000, border=2)
    trame_app.setInitCallback(lambda: trame_app.createAsyncTask(checkForStateUpdates(trame_app, bridge, view)))

    #
    #view.updateScale(trame_app.getImageScale())
    #

    trame_app.setStateValue('connected', False)
    trame_app.setStateValue('allow_submit', False)    

     # callback for steering enabled change
    def uiStateEnableSteeringUpdate(enable_steering, **kwargs):
        if trame_app.getStateValue('connected') is True:
            trame_app.setStateValue('allow_submit', enable_steering)
        if not enable_steering:
            bridge.sendUpdate({})

    # callback for color map change
    def uiStateColorMapUpdate(color_map, **kwargs):
        view.setColormap(color_map.lower())
        trame_app.pushFrame()
        
    # callback for barrier file upload
    """
    MonaLisa_Barrier.png 602x924
    TODO: make LBM-CFD simulation at that resolution (with flow from top to bottom)
    """
    def uiStateBarrierFromImage(barrier_file, **kwargs):
        if isinstance(barrier_file, dict):
            root, ext = os.path.splitext(barrier_file['name'])
            if ext.lower() in ['.png', '.jpeg', '.jpg', '.bmp']:
                file_content = np.frombuffer(barrier_file['content'], np.uint8)
                barrier_img = cv2.imdecode(file_content, cv2.IMREAD_UNCHANGED)
                barrier_new = None
                if len(barrier_img.shape) == 2: # grayscale
                    barrier_new = barrier_img.astype(bool)
                elif len(barrier_img.shape) == 3: # color
                    barrier_new = np.any(barrier_img != 0, axis=2)
                else:
                    print('Warning: barrier file not recognized as grayscale or color image')
                height, width = barrier_new.shape
                img_w, img_h = view.getSize()
                if width == img_w and height == img_h:
                    horizontal_barriers = findTrueRuns(barrier_new)
                    view.setBarriers(horizontal_barriers)
                else:
                    print(f'Warning: barrier image size {width}x{height} does not match simulation dimensions {img_w}x{img_h}')
            else:
                print('Warning: barrier file must be JPEG, PNG, or BMP')

    # callback to clear barriers
    def uiClearBarriers():
        view.clearBarriers()
        trame_app.pushFrame()

    # callback for clicking submit button
    def uiSubmitSteeringOptions():
        steering_data = {
            'flow_speed': float(trame_app.getStateValue('flow_speed')),
            'barriers': view.getBarriers()
        }
        bridge.sendUpdate(steering_data)

    # set up custom widgets
    widgets = [
        TDivider(),
        TSwitch('Enable Steering', 'enable_steering', True, on_change=uiStateEnableSteeringUpdate),
        TSpacer(),
        TSlider('Flow Speed', 'flow_speed', 0.25, 1.5, 0.05, 0.75),
        TDropDownMenu('Color Map', 'color_map', ['Divergent', 'Turbo', 'Inferno', 'StarryNight'], 'Divergent', on_change=uiStateColorMapUpdate),
        TSpacer(),
        TFileInput('Barrier Image', 'barrier_file', False, on_change=uiStateBarrierFromImage),
        TSpacer(),
        TButton('Clear Barriers', style='secondary', on_click=uiClearBarriers),
        TSpacer(),
        TButton('Submit', style='primary', disable='!allow_submit', on_click=uiSubmitSteeringOptions)
    ]
    trame_app.setPageLayout('Ascent-Trame', widgets)

    trame_app.start()


# Asynchronously check for state updates from Ascent
async def checkForStateUpdates(trame_app, bridge, view):
    while True:
        ascent_data = bridge.pollForPublishedData()
        if ascent_data is not None:
            trame_app.setStateValue('connected', True)

            if trame_app.getStateValue('enable_steering') is True:
                trame_app.setStateValue('allow_submit', True)

            view.updateData(ascent_data)
            trame_app.pushFrame()
            #view.updateScale(trame_app.getImageScale())

            if trame_app.getStateValue('enable_steering') is False:
               bridge.sendUpdate({}) 

        await asyncio.sleep(0)


# Find runs of True values in an array
def findTrueRuns(arr):
    # Step 1: Pad with False on both sides of each row
    padded = np.pad(arr, ((0, 0), (1, 1)), mode='constant', constant_values=False)
    
    # Step 2: Compute difference to find rising/falling edges
    diff = np.diff(padded.astype(np.int8), axis=1)
    
    # Step 3: Where diff == 1 -> start of True run, diff == -1 -> end of True run
    run_starts = np.argwhere(diff == 1)
    run_ends = np.argwhere(diff == -1)
    
    # Step 4: Combine into structured output: (row, start, row, end)
    # Assumes starts and ends match in order for each row
    assert np.array_equal(run_starts[:, 0], run_ends[:, 0]), "Mismatch in start/end rows"
    
    row_indices = run_starts[:, 0]
    run_data = np.column_stack((run_starts[:, 1], row_indices, run_ends[:, 1] - 1, row_indices))
    
    return run_data.astype(np.int32)

# Trame Custom View
class LbmCfdView(TrameImageView):
    def __init__(self):
        super().__init__()

        self._data = None
        #self._data = {'vorticity': np.zeros((924, 602), dtype=np.float32), 'barriers': np.empty(shape=(0,0), dtype=np.int32)}
        self._scale = 1.0
        self._base_image = None
        self._image = np.zeros((1, 2, 3), dtype=np.uint8)
        #self._image = np.zeros((924, 602, 3), dtype=np.uint8)
        self._jpeg_quality = 94
        self._colormaps = {
            'divergent': self._loadColorMap('../resrc/colormap_divergent.png'),
            'turbo': self._loadColorMap('../resrc/colormap_turbo.png'),
            'inferno': self._loadColorMap('../resrc/colormap_inferno.png'),
            'starrynight': self._loadColorMap('../resrc/colormap_starrynight.png')
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
        if self._data is not None:
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
            self.setFrameTime()
            return encoded_img
        return None

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
    Set barriers to new configuration
    """
    def setBarriers(self, barriers):
        if self._data is not None:
            self._data['barriers'] = barriers
            self._renderBarriers()
    
    """
    Clear all barriers
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
        mx = mouse_x #int(mouse_x / self._scale)
        my = height - mouse_y #height - int(mouse_y / self._scale)
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

