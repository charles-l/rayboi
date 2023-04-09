import pyglet
import numpy as np
import pyinotify # type: ignore
import importlib
import time
import os
import traceback

img = None
m = importlib.import_module('rayboi')
rayboi = None

cam_orig = [0, 0, 0]

def reload_and_render():
    global rayboi
    try:
        importlib.reload(m)
        print('import/compile opencl code...')
        rayboi = m.rayboi()
        print('done')
    except:
        traceback.print_exc()
        #import pdb; pdb.post_mortem()
    print("exit")

reload_and_render()

window = pyglet.window.Window(width=800, height=600)
key_handler = pyglet.window.key.KeyStateHandler()
window.push_handlers(key_handler)

fb = None
i = 1
while True:
    window.switch_to()
    pyglet.clock.tick()
    window.dispatch_events()
    window.clear()
    if rayboi is not None:
        print('start render...')
        newfb = rayboi.main(i).get().clip(0, 1)
        if fb is None:
            fb = newfb
        else:
            fb += newfb
        print('render complete')
        img = pyglet.image.ImageData(fb.shape[1], fb.shape[0], 'RGB', ((fb / i) * 255).astype(np.uint8).tobytes())
    if img is not None:
        img.blit(0, 0)

    time.sleep(0.05)

    window.flip()
    i += 1

pyglet.app.run()
