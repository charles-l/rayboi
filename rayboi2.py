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

    def ray_intersect(self, orig, direction) -> Tuple[np.ndarray, np.ndarray]:
        '''
        :returns: boolean array of hits, array of distances (floats)
        '''
        L = self.center - orig
        tca = v3_dots(L.reshape((-1, 3)), direction.reshape((-1, 3)))
        d2 = v3_dots(L, L) - tca**2
        # early check would normally check d2 > self.radius**2
        # FIXME: I don't think nans should be floating around here...
        thc = np.sqrt(self.radius**2 - d2)
        t0 = tca - thc
        t1 = tca + thc
        t0[t0 < 0] = t1[t0 < 0]
        return (np.where(d2 > self.radius**2, False,
                        np.where(t0 < 0, False, True)).ravel(),
                t0.ravel())


def normalize(v):
    return v / np.linalg.norm(v, axis=1)[:,np.newaxis]


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


def v3_dots(a, b):
    '''
    perform a dot product across the vector3s in a and b.
    e.g. v3_dots([v1, v2, v3], [v4, v5, v6]) => [v1 @ v4, v2 @ v5, v3 @ v6]
    '''
    assert len(a.shape) == len(b.shape) == 2
    assert a.shape[0] == b.shape[0] or (1 in (a.shape[0], b.shape[0])), f"can't broadcast a and b: {a.shape=}, {b.shape=}"
    assert a.shape[1] == b.shape[1] == 3, (a.shape, b.shape)
    return np.sum(a * b, axis=1)


def render():
    width, height = 600, 400
    fov = np.pi/3

    framebuffer = np.zeros((height * width, 3))

    spheres = [
        Sphere(np.array((-3, 0, -16)), 2),
        Sphere(np.array((1, -1.5, -12)), 2),
        Sphere(np.array((1.5, -0.5, -18)), 3),
        Sphere(np.array((7, 5, -18)), 4)
        ]

    sphere_centers = np.array([s.center for s in spheres])
    sphere_materials = np.array(
        # albedo      color            spec_exponent
        [((0.6, 0.3), (0.4, 0.4, 0.3), 50),
         ((0.9, 0.1), (0.3, 0.1, 0.1), 10),
         ((0.9, 0.1), (0.3, 0.1, 0.1), 10),
         ((0.6, 0.3), (0.4, 0.4, 0.3), 50),
         ],
        dtype=[('albedo', 'f4', 2),
               ('diffuse_color', 'f4', 3),
               ('specular_exponent', 'f4')])

    start = time.time()

    X, Y = np.meshgrid(np.arange(width), np.arange(height))
    xs =  (2*(X+0.5) / width - 1) * np.tan(fov/2) * width/height
    ys =  (2*(Y+0.5) / height - 1) * np.tan(fov/2)
    dirs = normalize(np.dstack((xs, ys, -1 * np.ones((height, width)))).reshape((-1, 3)))

    orig = np.array([[0, 0, 0]])
    dists, sphere_map = scene_intersect(orig, dirs, spheres)
    points = orig + dists.reshape((-1, 1)) * dirs

    N = normalize(points - sphere_centers[sphere_map])

    light_positions = np.array([(-20, 20, 20), (30, 50, -25), (30, 20, 30)])
    light_intensities = np.array([1.5, 1.8, 1.7])

    diffuse_intensity = np.zeros((width*height,), dtype=np.float64)
    specular_intensity = np.zeros((width*height,), dtype=np.float64)

    for i in range(len(light_positions)):
        light_dir = normalize(light_positions[i] - points)
        light_distance = np.linalg.norm(light_positions[i] - points, axis=1)

        shadow_orig = np.where((v3_dots(light_dir.reshape((-1, 3)), N) < 0).reshape((-1, 1)),
                               points - N * 1e-3,
                               points + N * 1e-3)

        shadow_dists, shadow_map = scene_intersect(shadow_orig, light_dir, spheres)
        shadow_points = orig + shadow_dists.reshape((-1, 1)) * light_dir
        shadow_mask = shadow_map != -1 # & (np.linalg.norm(shadow_points - shadow_orig, axis=1) < light_distance)

        d = v3_dots(light_dir, N)
        diffuse_intensity += light_intensities[i] * np.where((d < 0) | shadow_mask, 0, d)

        reflect = light_dir - 2 * N * v3_dots(light_dir, N).reshape((-1, 1))
        reflect_dot_dir = v3_dots(reflect, dirs)
        spec_exponents = sphere_materials['specular_exponent'][np.where(sphere_map != -1, sphere_map, 0)]
        specular_intensity += light_intensities[i] * np.power(np.where((reflect_dot_dir < 0) | shadow_mask, 0, reflect_dot_dir), spec_exponents)

    # clear to bg color
    framebuffer[:,:] = (0.2, 0.7, 0.8)

    framebuffer[sphere_map != -1] = (
        sphere_materials['diffuse_color'][sphere_map[sphere_map != -1]].reshape((-1, 3))
        * (diffuse_intensity[sphere_map != -1]
           * sphere_materials['albedo'][sphere_map[sphere_map != -1]][:,0]).reshape((-1, 1)) +
        (np.array((1, 1, 1)).reshape((3, 1)) * specular_intensity[sphere_map != -1]).T
        * sphere_materials['albedo'][sphere_map[sphere_map != -1]][:,1].reshape((-1, 1))
        )

    end = time.time()
    print(f'took {end-start:.2f}s')

    # HACK: not sure if this normalization is technically correct
    max_channel = framebuffer.max(axis=1)
    framebuffer[max_channel > 1] /= max_channel[max_channel > 1].reshape((-1, 1))
    framebuffer[framebuffer < 0] = 0
    framebuffer[framebuffer > 1] = 1
    return framebuffer.reshape((height, width, 3))
