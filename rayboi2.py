# TODO: build debugger interface with pyglet that autodetects changes and
#       rerenders, and allows you to switch to different debug views
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

    def ray_intersect(self, orig, direction) -> Tuple[np.ndarray, np.ndarray]:
        '''
        :returns: boolean array of hits, array of distances (floats)
        '''
        L = self.center - orig
        tca = direction @ L
        d2 = L @ L - tca**2
        # early check would normally check d2 > self.radius**2
        # FIXME: I don't think nans should be floating around here...
        thc = np.sqrt(self.radius**2 - d2)
        t0 = tca - thc
        t1 = tca + thc
        t0[t0 < 0] = t1[t0 < 0]
        return (np.where(d2 > self.radius**2, False,
                        np.where(t0 < 0, False, True)),
                t0)

def normalize(v):
    return v / np.linalg.norm(v)


def scene_intersect(orig, dirs, spheres):
    sphere_dist = np.inf * np.ones((dirs.shape[0]))
    sphere_mask = -1 * np.ones_like(sphere_dist)

    for i, sphere in enumerate(spheres):
        mask, dists = sphere.ray_intersect(orig, dirs)
        sphere_mask[mask & (dists < sphere_dist)] = i
        sphere_dist[mask & (dists < sphere_dist)] = dists[mask & (dists < sphere_dist)]

    # TODO: determine if I need this
    sphere_mask[sphere_dist > 1000] = -1

    return sphere_dist, sphere_mask.astype(np.int8)


def render():
    width, height = 1024, 768
    fov = np.pi/2

    framebuffer = np.zeros((height * width, 3))

    spheres = [
        Sphere(np.array((-3, 0, -15)), 2, (0.1, 0.1, 0.3)),
        Sphere(np.array((-3, 2, -16)), 2, (0.4, 0.6, 0.3))
        ]

    sphere_centers = np.array([s.center for s in spheres])
    sphere_colors = np.array([s.color for s in spheres]).reshape((-1, 3))

    start = time.time()

    X, Y = np.meshgrid(np.arange(width), np.arange(height))
    xs =  (2*(X+0.5) / width - 1) * np.tan(fov/2) * width/height
    ys = -(2*(Y+0.5) / height - 1) * np.tan(fov/2)
    dirs = np.dstack((xs, ys, -1 * np.ones((height, width)))).reshape((-1, 3))
    dirs = dirs / np.linalg.norm(dirs, axis=1).reshape((-1, 1))

    orig = np.array([0, 0, 0])
    dists, sphere_map = scene_intersect(orig, dirs, spheres)
    hits = orig + dists.reshape((-1, 1)) * dirs

    N = (hits - sphere_centers[sphere_map])

    light_positions = np.array([(20, 10, 10), (-20, 10, -10)])
    light_intensities = np.array([1, 0.5])
    light_totals = np.zeros((width*height,), dtype=np.float64)

    for i in range(len(light_positions)):
        light_dir = (light_positions[i] - hits).reshape((-1, 3))
        light_dir = light_dir / np.linalg.norm(light_dir, axis=1).reshape((-1, 1))
        d = np.sum(light_dir * N, axis=1)
        light_totals += light_intensities[i] * np.where(d < 0, 0, d)

    # clear to bg color
    framebuffer[:,:] = (0.2, 0.7, 0.8)

    color_map = sphere_colors[sphere_map[sphere_map != -1]]
    framebuffer[sphere_map != -1] = (light_totals[sphere_map != -1].reshape((-1, 1))) * color_map

    end = time.time()
    print(f'took {end-start:.2f}s')

    # HACK: this clamping throws away useful info
    framebuffer[framebuffer < 0] = 0
    framebuffer[framebuffer > 1] = 1
    return framebuffer.reshape((height, width, 3))
