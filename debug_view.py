import pyglet
import numpy as np
import importlib
import time
import os
import traceback

img = None
m = importlib.import_module('rayboi')
rayboi = None

cam_orig = [0, 0, 0]

def parse_obj(filename):
    objs = {}
    verts = []
    curobject = None
    with open(filename, 'r') as f:
        while (line := f.readline()):
            tag, *rest = line.strip().split(' ')
            if tag == '#':
                # comment
                pass
            elif tag == 'o':
                curobject = rest[0]
                objs[curobject] = []
                verts = []
            elif tag == 'v':
                verts.append([float(x) for x in rest])
            elif tag == 'f':
                ixs = [int(x) - 1 for x in rest]
                objs[curobject].append([verts[ixs[0]], verts[ixs[1]], verts[ixs[2]]])
            else:
                print('skipping unknown tag', tag)
    return objs

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
label = pyglet.text.Label('Iteration count...',
                          color=(0, 255, 0, 255),
                          font_size=12,
                          x=10, y=10,
                          anchor_x='left', anchor_y='bottom')

log = pyglet.text.layout.ScrollableTextLayout(pyglet.text.document.UnformattedDocument(''),
        width=400, height=200,
        multiline=True)
log.x = 0
log.y = 600
log.anchor_y = "top"
log.document.styles['color'] = (255, 0, 255, 255)

def printlog(msg):
    log.document.text += msg + '\n'

icosphere = np.array(parse_obj('icosphere.obj')['Icosphere']).astype(np.float32)
icosphere[:, :, 2] -= 10

fb = None
i = 1
while True:
    if key_handler[pyglet.window.key.ESCAPE]:
        break

    if key_handler[pyglet.window.key.S]:
        img.save(f'render-{i}.png')
        printlog(f'saved image render-{i}.png')

    window.switch_to()
    pyglet.clock.tick()
    window.dispatch_events()
    window.clear()

    if rayboi is not None:
        print('start render...')
        start = time.perf_counter_ns()
        renderfb = rayboi.main(i, icosphere)
        render_time = time.perf_counter_ns() - start
        newfb = renderfb.get().clip(0, 1)
        if fb is None:
            fb = newfb
        else:
            fb += newfb
        print('render complete')
        img = pyglet.image.ImageData(fb.shape[1], fb.shape[0], 'RGB', ((fb / i) * 255).astype(np.uint8).tobytes())
    if img is not None:
        img.blit(0, 0)


    log.view_y = -log.content_height
    log.draw()
    label.draw()
    pyglet.clock.get_default().sleep(500)

    window.flip()
    i += 1
    label.text = f'Iteration {i}. Time for frame: {render_time/1000/1000:.2f} ms'


window.close()
