import pyglet
import numpy as np
import pyinotify # type: ignore
import importlib
import time
import os
import traceback

img = None
m = importlib.import_module('rayboi')
new_img = None

cam_orig = [0, 0, 0]

def reload_and_render():
    global new_img
    try:
        importlib.reload(m)
        fb = m.rayboi().main(*cam_orig)
        print(fb)

        new_img = pyglet.image.ImageData(fb.shape[1], fb.shape[0], 'RGB', (fb * 255).astype(np.uint8).get().tobytes())
    except:
        traceback.print_exc()
        #import pdb; pdb.post_mortem()
    print("exit")

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

window = pyglet.window.Window(width=800, height=600)
key_handler = pyglet.window.key.KeyStateHandler()
window.push_handlers(key_handler)

while True:
    window.switch_to()
    pyglet.clock.tick()
    window.dispatch_events()
    window.clear()
    if new_img:
        img = new_img
        new_img = None
    if img:
        img.blit(0, 0)

    window.flip()
    time.sleep(0.1)

pyglet.app.run()
file_watcher.stop()
