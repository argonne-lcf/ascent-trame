from trame.app import get_server, asynchronous
from trame.widgets import vuetify, rca, client
from trame.ui.vuetify import SinglePageLayout

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
        self._image_scale = 1.0

        if self._fixed_width > 0:
            self._state.vis_style = f'width: {self._fixed_width}px; height: {self._fixed_width // 2}px; border: solid {self._border_size}px #000000; box-sizing: content-box;'
        else:
            self._state.vis_style = f'border: solid {self._border_size}px #000000; box-sizing: content-box;'

        # register RCA view with Trame controller
        self._view_handler = None
        @self._ctrl.add("on_server_ready")
        def initRca(**kwargs):
            #nonlocal view_handler
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

    def getImageScale(self):
        return self._image_scale

    def setPageLayout(self, title, widgets):
        for w in widgets:
            if (w['type'] == 'switch' or w['type'] == 'slider' or w['type'] == 'dropdown') and 'on_change' in w:
                self._state.change(w['state_var'])(w['on_change'])

        with SinglePageLayout(self._server) as layout:
            if self._fixed_width > 0:
                client.Style('#rca-view div div img { width: 100%; height: auto; }')
            layout.title.set_text(title)
            with layout.toolbar:
                for w in widgets:
                    if w['type'] == 'divider':
                        vuetify.VDivider(vertical=True, classes="mx-2")
                    elif w['type'] == 'spacer':
                        vuetify.VSpacer()
                    elif w['type'] == 'switch':
                        vuetify.VSwitch(
                            label=w['label'],
                            v_model=(w['state_var'], w['value']),
                            hide_details=True,
                            dense=True
                        )
                    elif w['type'] == 'text':
                        text = w['value']
                        if 'is_state_var' in w and w['is_state_var'] is True:
                            if 'float_digits' in w and isinstance(w['float_digits'], int):
                                text = f'{{{{{w["value"]}.toFixed({w["float_digits"]})}}}}'
                            else:
                                text = f'{{{{{w["value"]}}}}}'
                        vuetify.VCol(text)
                    elif w['type'] == 'button':
                        if 'disable' in w:
                            vuetify.VBtn(
                                w['value'],
                                color=w['style'],
                                disabled=(w['disable'],),
                                click=w['on_click']
                            )
                        else:
                            vuetify.VBtn(
                                w['value'],
                                color=w['style'],
                                click=w['on_click']
                            )
                    elif w['type'] == 'slider':
                        vuetify.VSlider(
                            label=w['label'],
                            v_model=(w['state_var'], w['value']),
                            min=w['min'],
                            max=w['max'],
                            step=w['step'],
                            hide_details=True,
                            dense=True
                        )
                    elif w['type'] == 'dropdown':
                        vuetify.VSelect(
                            label=w['label'],
                            v_model=(w['state_var'], w['value']),
                            items=(str(w['options']),),
                            hide_details=True,
                            dense=True
                        )
            with layout.content:
                with vuetify.VContainer(fluid=True, classes='pa-0 fill-height', style='justify-content: center; align-items: start;'):
                    v = rca.RemoteControlledArea(name='view', display='image', id='rca-view', style=('vis_style',))

    def pushFrame(self):
        if self._view_handler is not None:
            if self._fixed_width > 0:
                w, h = self._view_handler.getImageSize()
                img_h = self._fixed_width * h // w
                self._image_scale = self._fixed_width / w
                self._state.update({'vis_style': f'width: {self._fixed_width}px; height: {img_h}px; border: solid {self._border_size}px #000000; box-sizing: content-box;'})
                self._state.flush()
            self._view_handler.pushFrame()

    def start(self):
        self._server.start()


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

    def getImageSize(self):
        return self._view.getSize()

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

