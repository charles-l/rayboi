import numpy as np
import numpy.ma as ma
from PIL import Image
from dataclasses import dataclass
import time
from typing import Tuple

width, height = 600, 400

v3_dtype = np.dtype(('f4', 3))


@dataclass
class Sphere:
    center: np.ndarray
    radius: float

    def ray_intersect(self, orig, direction) -> Tuple[np.ndarray, np.ndarray]:
        '''
        :returns: boolean array of points hit, array of distances (floats)
        '''
        # TODO: determine if it's more efficient to just return a list of
        # indices rather than generating a bunch of crap with the mask data.
        L = self.center - orig
        tca = v3_dots(L.reshape((-1, 3)), direction.reshape((-1, 3)))
        d2 = v3_dots(L, L) - tca**2
        # early check would normally check d2 > self.radius**2
        # FIXME: I don't think nans should be floating around here...
        thc = np.sqrt(self.radius**2 - d2)
        t0 = tca - thc
        t1 = tca + thc
        t0[t0 < 0] = t1[t0 < 0]
        mask = np.where(d2 > self.radius**2, False,
                        np.where(t0 < 0, False, True)).ravel()
        return ma.array(t0, mask=mask)


def normalize(v):
    return v / np.linalg.norm(v, axis=1)[:, np.newaxis]


def scene_intersect(orig, dirs, spheres):
    sphere_dist = np.inf * np.ones((dirs.shape[0]))
    sphere_mask = -1 * np.ones_like(sphere_dist)

    for i, sphere in enumerate(spheres):
        dists = sphere.ray_intersect(orig, dirs)
        sphere_mask[dists.mask & (dists < sphere_dist)] = i
        sphere_dist[dists.mask & (dists < sphere_dist)
                    ] = dists[dists.mask & (dists < sphere_dist)]

    # TODO: determine if I need this
    sphere_mask[sphere_dist > 1000] = -1

    return sphere_dist, sphere_mask.astype(np.int8)


def v3_dots(a, b):
    '''
    perform a dot product across the vector3s in a and b.
    e.g. v3_dots([v1, v2, v3], [v4, v5, v6]) => [v1 @ v4, v2 @ v5, v3 @ v6]
    '''
    assert len(a.shape) == len(b.shape) == 2
    assert a.shape[0] == b.shape[0] or (1 in (
        a.shape[0], b.shape[0])), f"can't broadcast a and b: {a.shape=}, {b.shape=}"
    assert a.shape[1] == b.shape[1] == 3, (a.shape, b.shape)
    return np.sum(a * b, axis=1)


bg_color = (0.2, 0.7, 0.8)


def reflect(I, N):
    return I - 2 * N * v3_dots(I, N).reshape((-1, 1))


def refract(I, N, refractive_index):
    '''Snell's law'''
    cosi = -np.clip(v3_dots(I, N), -1, 1).reshape((-1, 1))
    etai = np.ones_like(cosi)
    etat = refractive_index.reshape((-1, 1)) * np.ones_like(cosi)
    n = N * np.ones_like(cosi)

    swap_mask = cosi < 0

    cosi[swap_mask] = -cosi[swap_mask]
    etai[swap_mask], etat[swap_mask] = etat[swap_mask], etai[swap_mask]
    n[swap_mask.ravel()] = -n[swap_mask.ravel()]

    eta = etai / etat
    k = 1 - eta**2 * (1 - cosi**2)
    return np.where(k < 0, (0, 0, 0), I * eta + n * (eta * cosi - np.sqrt(k)))


def cast_rays(origs, dirs, spheres, lights, n_bounces=3):
    sphere_centers = np.array([s.center for s in spheres])
    #    ior   albedo                 color            spec_exponent
    sphere_materials = np.array(
        [(1.0, (0.6, 0.3, 0.1, 0.0),  (0.4, 0.4, 0.3), 50),
         (1.5, (0.0, 0.5, 0.1, 0.8),  (0.6, 0.7, 0.8), 125),
         (1.0, (0.9, 0.1, 0.0, 0.0),  (0.3, 0.1, 0.1), 10),
         (1.0, (0.0, 10.0, 0.8, 0.0), (1.0, 1.0, 1.0), 1425)
         ],
        dtype=[
            ('ior', 'f4'),
            ('albedo', 'f4', 4),
            ('diffuse_color', 'f4', 3),
            ('specular_exponent', 'f4')])

    dists, object_map = scene_intersect(origs, dirs, spheres)

    # points are filtered down only to rays that hit anything
    points = (origs + dists.reshape((-1, 1)) * dirs)[object_map != -1]
    hit_object_map = object_map[object_map != -1]
    hit_origs = origs[object_map != -1] if len(origs) > 1 else origs
    hit_dirs = dirs[object_map != -1]

    N = normalize(points - sphere_centers[hit_object_map])

    diffuse_intensity = np.zeros_like(points, dtype=np.float64)
    specular_intensity = np.zeros_like(points, dtype=np.float64)

    for i in range(len(lights)):
        light_dir = normalize(lights['position'][i] - points).reshape((-1, 3))
        light_distance = np.linalg.norm(lights['position'][i] - points, axis=1)

        shadow_orig = np.where((v3_dots(light_dir, N) < 0).reshape((-1, 1)),
                               points - N * 1e-3,
                               points + N * 1e-3)

        shadow_dists, shadow_map = scene_intersect(
            shadow_orig, light_dir, spheres)
        shadow_points = hit_origs + shadow_dists.reshape((-1, 1)) * light_dir
        shadow_mask = (
            (shadow_map != -1) &
            (np.linalg.norm(shadow_points - shadow_orig, axis=1) < light_distance))

        d = v3_dots(light_dir, N)
        diffuse_intensity += np.where((d < 0) | shadow_mask,
                                      0, lights['intensity'][i] * d)[:, np.newaxis]

        spec_exponents = sphere_materials['specular_exponent'][hit_object_map]

        specular_intensity += (
            (shadow_map == -1) *
            lights['intensity'][i] *
            np.clip(v3_dots(reflect(light_dir, N), hit_dirs), 0, None) ** spec_exponents)[:, np.newaxis]

    reflect_dir = normalize(reflect(hit_dirs, N))
    refract_dir = refract(
        hit_dirs, N, sphere_materials['ior'][hit_object_map])
    reflect_origs = np.where((v3_dots(reflect_dir, N) < 0)[
                             :, np.newaxis], points - N*1e-3, points + N*1e-3)
    refract_origs = np.where((v3_dots(refract_dir, N) < 0)[
                             :, np.newaxis], points - N*1e-3, points + N*1e-3)
    if n_bounces == 0:
        refract_colors = reflect_colors = np.ones_like(
            reflect_origs) * bg_color
    else:
        reflect_colors = cast_rays(
            reflect_origs, reflect_dir, spheres, lights, n_bounces-1)
        refract_colors = cast_rays(
            refract_origs, refract_dir, spheres, lights, n_bounces-1)

    r = np.zeros_like(dirs)
    r[:] = bg_color

    r[object_map != -1] = (
        # diffuse
        sphere_materials['diffuse_color'][hit_object_map]
        * (diffuse_intensity * sphere_materials['albedo'][hit_object_map][:, 0, np.newaxis])

        # specular
        + (np.array(((1, 1, 1))) * specular_intensity) *
        sphere_materials['albedo'][hit_object_map][:, 1, np.newaxis]

        # reflection
        + reflect_colors *
        sphere_materials['albedo'][hit_object_map][:, 2, np.newaxis]

        # refraction
        + refract_colors *
        sphere_materials['albedo'][hit_object_map][:, 3, np.newaxis]
    )

    return r


def render():
    fov = np.pi/3

    framebuffer = np.zeros((height * width, 3))

    spheres = [
        Sphere(np.array((-3, 0, -16)), 2),
        Sphere(np.array((1, -1.5, -12)), 2),
        Sphere(np.array((1.5, -0.5, -18)), 3),
        Sphere(np.array((7, 5, -18)), 4)
    ]

    start = time.time()

    X, Y = np.meshgrid(np.arange(width), np.arange(height))
    xs = (2*(X+0.5) / width - 1) * np.tan(fov/2) * width/height
    ys = (2*(Y+0.5) / height - 1) * np.tan(fov/2)
    dirs = normalize(
        np.dstack((xs, ys, -1 * np.ones((height, width)))).reshape((-1, 3)))

    orig = np.array([(0, 0, 0)], dtype=v3_dtype)
    lights = np.array(
        [((-20, 20, 20), 1.5),
         ((30, 50, -25), 1.8),
         ((30, 20, 30), 1.7)],
        dtype=[('position', 'f4', 3), ('intensity', 'f4')])

    # clear to bg color
    framebuffer[:, :] = cast_rays(orig, dirs, spheres, lights)

    end = time.time()
    print(f'took {end-start:.2f}s')

    # HACK: not sure if this normalization is technically correct
    max_channel = framebuffer.max(axis=1)
    framebuffer[max_channel >
                1] /= max_channel[max_channel > 1].reshape((-1, 1))
    framebuffer[framebuffer < 0] = 0
    framebuffer[framebuffer > 1] = 1
    return framebuffer.reshape((height, width, 3))
