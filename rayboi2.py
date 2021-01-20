import numpy as np
from PIL import Image
from dataclasses import dataclass
import time
from typing import Tuple

@dataclass
class Sphere:
    center: np.ndarray
    radius: float
    color: Tuple[int, int, int]

    def ray_intersect(self, orig, direction, t0):
        L = self.center - orig
        tca = direction @ L
        d2 = L @ L - tca**2
        # early check would normally check d2 > self.radius**2
        thc = np.sqrt(self.radius**2 - d2)
        t0 = tca - thc
        t1 = tca + thc
        return np.where(d2 > self.radius**2, False,
                        np.where(t0 < 0, np.where(t1 < 0, False, True), True))

def render():
    width, height = 1024, 768
    fov = np.pi/2

    framebuffer = np.zeros((height, width, 3))

    sphere = Sphere(np.array((-3, 0, -16)), 2, (0.4, 0.4, 0.3))

    start = time.time()

    X, Y = np.meshgrid(np.arange(width), np.arange(height))
    xs =  (2*(X+0.5) / width - 1) * np.tan(fov/2) * width/height
    ys = -(2*(Y+0.5) / height - 1) * np.tan(fov/2)
    ps = np.dstack((xs, ys, -1 * np.ones((height, width)))).reshape((-1, 3))
    ds = ps / np.linalg.norm(ps, axis=1).reshape((-1, 1))

    framebuffer[:,:] = (0.2, 0.7, 0.8)
    framebuffer[sphere.ray_intersect(np.array((0, 0, 0)), ds, np.inf).reshape((height, width))] = sphere.color

    end = time.time()
    print(f'took {end-start:.2f}s')

    Image.fromarray((framebuffer * 255).astype(np.uint8)).save('render.png')

render()
