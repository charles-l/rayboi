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

    return sphere_dist < 1000, sphere_mask.astype(np.int8)


def render():
    width, height = 1024, 768
    fov = np.pi/2

    framebuffer = np.zeros((height, width, 3))

    spheres = [
        Sphere(np.array((-3, 0, -15)), 2, (0.1, 0.1, 0.3)),
        Sphere(np.array((-3, 2, -16)), 2, (0.4, 0.6, 0.3))
        ]

    start = time.time()

    X, Y = np.meshgrid(np.arange(width), np.arange(height))
    xs =  (2*(X+0.5) / width - 1) * np.tan(fov/2) * width/height
    ys = -(2*(Y+0.5) / height - 1) * np.tan(fov/2)
    ps = np.dstack((xs, ys, -1 * np.ones((height, width)))).reshape((-1, 3))
    ds = ps / np.linalg.norm(ps, axis=1).reshape((-1, 1))

    framebuffer[:,:] = (0.2, 0.7, 0.8)
    dists, sphere_map = scene_intersect((0, 0, 0), ds, spheres)
    dists = dists.reshape((height, width))
    sphere_map = sphere_map.reshape((height, width))
    framebuffer[sphere_map != -1] = [spheres[i].color for i in sphere_map[sphere_map != -1]]
    #framebuffer[sphere.ray_intersect(np.array((0, 0, 0)), ds, np.inf).reshape((height, width))] = sphere.color

    end = time.time()
    print(f'took {end-start:.2f}s')

    return framebuffer
