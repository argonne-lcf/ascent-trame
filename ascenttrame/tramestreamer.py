import time
from trame.app import get_server, asynchronous
from trame.widgets import vuetify, rca, client
from trame.ui.vuetify import SinglePageLayout

# Trame widgets
class TDivider:
    def __init__(self):
        self.type = 'divider'

    def generateWidget(self):
        vuetify.VDivider(vertical=True, classes="mx-2")

class TSpacer:
    def __init__(self):
        self.type = 'spacer'

    def generateWidget(self):
        vuetify.VSpacer()

class TText:
    def __init__(self, value, is_state_var=False, float_digits=-1):
        self.type = 'text'
        self.value = value
        self.is_state_var = is_state_var
        self.float_digits = float_digits

    def generateWidget(self):
        text = self.value
        if is_state_var:
            if isinstance(self.float_digits, int) and self.float_digits >= 0:
                text = f'{{{{{self.value}.toFixed({self.float_digits})}}}}'
            else:
                text = f'{{{{{self.value}}}}}'
        vuetify.VCol(text)

class TSwitch:
    def __init__(self, label, state_var, value, on_change=None):
        self.type = 'switch'
        self.label = label
        self.state_var = state_var
        self.value = value
        self.on_change = on_change

    def generateWidget(self):
        vuetify.VSwitch(
            label=self.label,
            v_model=(self.state_var, self.value),
            hide_details=True,
            dense=True
        )

class TButton:
    def __init__(self, value, style='secondary', disable='', on_click=lambda: None):
        self.type = 'button'
        self.value = value
        self.disable = disable
        self.style = style
        self.on_click = on_click

    def generateWidget(self):
        if isinstance(self.disable, str) and len(self.disable) > 0:
            vuetify.VBtn(
                self.value,
                color=self.style,
                disabled=(self.disable,),
                click=self.on_click
            )
        else:
            vuetify.VBtn(
                self.value,
                color=self.style,
                click=self.on_click
            )
        
class TSlider:
    def __init__(self, label, state_var, vmin, vmax, step, value, on_change=None):
        self.type = 'slider'
        self.label = label
        self.state_var = state_var
        self.min = vmin
        self.max = vmax
        self.step = step
        self.value = value
        self.on_change = on_change

    def generateWidget(self):
        vuetify.VSlider(
            label=self.label,
            v_model=(self.state_var, self.value),
            min=self.min,
            max=self.max,
            step=self.step,
            hide_details=True,
            dense=True
        )

        text = f'{{{{{self.state_var}}}}}'
        if isinstance(self.step, float):
            step_str = str(self.step)
            integer, decimals = step_str.split('.')
            num_dec = len(decimals)
            text = f'{{{{{self.state_var}.toFixed({num_dec})}}}}'
        vuetify.VCol(text)

class TDropDownMenu:
    def __init__(self, label, state_var, options, value, on_change=None):
        self.type = 'dropdown'
        self.label = label
        self.state_var = state_var
        self.options = options
        self.value = value
        self.on_change = on_change

    def generateWidget(self):
        vuetify.VSelect(
            label=self.label,
            v_model=(self.state_var, self.value),
            items=(str(self.options),),
            hide_details=True,
            dense=True
        )


# Trame Image Streamer
class TrameImageStreamer:
    def __init__(self, view, fixed_width=0, border=0):
        # set up Trame application
        self._server = get_server(client_type="vue2")
        self._state = self._server.state
        self._ctrl = self._server.controller
        self._init_callback = None

        self._fixed_width = fixed_width
        self._border_size = border

        if self._fixed_width > 0:
            self._state.vis_style = f'width: {self._fixed_width}px; height: {self._fixed_width // 2}px; border: solid {self._border_size}px #000000; box-sizing: content-box;'
        else:
            self._state.vis_style = f'border: solid {self._border_size}px #000000; box-sizing: content-box;'

        # register RCA view with Trame controller
        self._view_handler = None
        @self._ctrl.add("on_server_ready")
        def initRca(**kwargs):
            self._view_handler = RcaViewAdapter(view, 'view')
            self._ctrl.rc_area_register(self._view_handler)
            if self._init_callback is not None:
                self._init_callback()

    def setInitCallback(self, callback):
        self._init_callback = callback

    def getStateValue(self, attr):
        return getattr(self._state, attr, None)

    def setStateValue(self, attr, value):
        setattr(self._state, attr, value)

    def createAsyncTask(self, async_func):
        asynchronous.create_task(async_func)

    def setPageLayout(self, title, widgets):
        for w in widgets:
            if getattr(w, 'on_change', None) is not None:
                self._state.change(w.state_var)(w.on_change)

        with SinglePageLayout(self._server) as layout:
            if self._fixed_width > 0:
                client.Style('#rca-view div div img { width: 100%; height: auto; }')
            layout.title.set_text(title)
            with layout.toolbar:
                for w in widgets:
                    w.generateWidget()
            with layout.content:
                with vuetify.VContainer(fluid=True, classes='pa-0 fill-height', style='justify-content: center; align-items: start;'):
                    v = rca.RemoteControlledArea(name='view', display='image', id='rca-view', style=('vis_style',))

    def pushFrame(self):
        if self._view_handler is not None:
            if self._fixed_width > 0:
                w, h = self._view_handler.getImageSize()
                img_h = self._fixed_width * h // w
                self._view_handler.setImageScale(self._fixed_width / w)
                self._state.update({'vis_style': f'width: {self._fixed_width}px; height: {img_h}px; border: solid {self._border_size}px #000000; box-sizing: content-box;'})
                self._state.flush()
            self._view_handler.pushFrame()

    def start(self):
        self._server.start()


# Trame RCA View (Base Class)
class TrameImageView:
    def __init__(self):
        self._frame_time = round(time.time_ns() / 1000000)

    def getSize(self):
        return (0, 0)

    def getFrame(self):
        return None

    def setFrameTime(self):
        self._frame_time = round(time.time_ns() / 1000000)

    def getFrameTime(self):
        return self._frame_time

    def onLeftMouseButton(self, mouse_x, mouse_y, pressed):
        print('TrameImageView: Left mouse')
        return False

    def onRightMouseButton(self, mouse_x, mouse_y, pressed):
        print('TrameImageView: Right mouse')
        return False

    def onMouseMove(self, mouse_x, mouse_y):
        return False


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
        self._image_scale = 1.0

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

    def getImageSize(self):
        return self._view.getSize()

    def setImageScale(self, scale):
        self._image_scale = scale

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
            rerender = self._view.onLeftMouseButton(int(event['x'] / self._image_scale), int(event['y'] / self._image_scale), True)
        elif event_type == 'LeftButtonRelease':
            rerender = self._view.onLeftMouseButton(int(event['x'] / self._image_scale), int(event['y'] / self._image_scale), False)
        if event_type == 'RightButtonPress':
            rerender = self._view.onRightMouseButton(int(event['x'] / self._image_scale), int(event['y'] / self._image_scale), True)
        elif event_type == 'RightButtonRelease':
            rerender = self._view.onRightMouseButton(int(event['x'] / self._image_scale), int(event['y'] / self._image_scale), False)
        elif event_type == 'MouseMove':
            rerender = self._view.onMouseMove(int(event['x'] / self._image_scale), int(event['y'] / self._image_scale))

        if rerender:
            frame_data = self._view.getFrame()
            self._streamer.push_content(self.area_name, self._getMetadata(), frame_data.data)

