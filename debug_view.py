import pyglet
import numpy as np
import pyinotify # type: ignore
import importlib
import time
import traceback

img = None
m = importlib.import_module('rayboi2')

def reload_and_render():
    global img
    try:
        importlib.reload(m)
        fb = m.render()

        '''
        # FIXME: add debug and do proper normalize
        dfb = np.where(m.debug_views[0] == np.inf, 0, m.debug_views[0])
        dfb = dfb - dfb.min()
        print(dfb.min(), dfb.max())
        dfb = (dfb / dfb.max()).reshape((fb.shape[1], fb.shape[0]))
        dfb = np.dstack((dfb, dfb, dfb))
        '''

        img = pyglet.image.ImageData(fb.shape[1], fb.shape[0], 'RGB', (fb * 255).astype(np.uint8).tobytes())
    except:
        traceback.print_exc()
        #import pdb; pdb.post_mortem()

class OnWriteHandler(pyinotify.ProcessEvent):
    def process_IN_MODIFY(self, event):
        print('==> Modification detected', event.pathname)
        reload_and_render()
        pyglet.app.platform_event_loop.notify()

watch_manager = pyinotify.WatchManager()
watch_manager.add_watch('.', pyinotify.IN_MODIFY)
file_watcher = pyinotify.ThreadedNotifier(watch_manager, OnWriteHandler())
file_watcher.start()

reload_and_render()

window = pyglet.window.Window(width=m.width, height=m.height)

while True:
    window.switch_to()
    pyglet.clock.tick()
    window.dispatch_events()
    window.clear()
    if img:
        img.blit(0, 0)

    window.flip()
    time.sleep(1)

pyglet.app.run()
file_watcher.stop()
