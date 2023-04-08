import "lib/github.com/diku-dk/cpprandom/random"

def width: i64 = 800
def height: i64 = 600
def fov: f32 = f32.pi / 3
def samples_per_ray: i64 = 100

type Vec = {x: f32, y: f32, z: f32}
type Ray = {origin: Vec, dir: Vec}
type Sphere = { pos: Vec, radius: f32, color: Vec, emission: Vec }

def toarr (a: Vec) = [a.x, a.y, a.z]
def f32toarr(f: f32) = [f, f, f]

def (a: Vec) <+> (b: Vec): Vec = {x = a.x + b.x, y = a.y + b.y, z = a.z + b.z}
def (a: Vec) <-> (b: Vec): Vec = {x = a.x - b.x, y = a.y - b.y, z = a.z - b.z}
def (s: f32) <*> (b: Vec): Vec = {x = s * b.x, y = s * b.y, z = s * b.z}
def dot (a: Vec) (b: Vec): f32 = (a.x*b.x) + (a.y*b.y) + (a.z*b.z)
def normalize(a: Vec): Vec = (1 / (dot a a |> f32.sqrt)) <*> a
def vec a b c = {x=a, y=b, z=c}
def vecavg vs = (1/(length vs |> f32.i64)) <*> (reduce (<+>) (vec 0 0 0) vs)

module randfloat = uniform_real_distribution f32 minstd_rand

def rand_sphere_point rng =
    let (rng, u) = (randfloat.rand (0, 1) rng) in
    let (rng, v) = (randfloat.rand (0, 1) rng) in
    let theta = 2 * f32.pi * u in
    let phi = f32.acos (2 * v - 1) in
    (rng, {x = f32.cos theta * f32.sin phi, y = f32.sin theta * f32.sin phi, z = f32.cos phi})

def imin [n] (arr: [n]f32): i64 =
    let (i, _) = reduce (\(ai, ax) (i, x) -> (if x < ax then (i, x) else (ai, ax))) (-1, f32.inf) (zip (iota n) arr) in
    i

def check_sphere (ray: Ray) ({pos, radius, color, emission}: Sphere): f32 =
    let ray_origin = ray.origin <-> pos in
    let s = f32.sqrt(((dot ray_origin ray.dir) * (dot ray_origin ray.dir)) - ((dot ray_origin ray_origin) - radius*radius)) in
    let d1 = (dot (-1 <*> ray_origin) ray.dir) + s in
    let d2 = (dot (-1 <*> ray_origin) ray.dir) - s in
    let d = f32.min d1 d2 in
    if f32.isnan d || d < 0 then f32.inf else d

def trace_ray rng (init_ray: Ray) (spheres: []Sphere) =
    let (_, final_color, final_light, rng) = loop (ray, color, light, rng) = (init_ray, {x=1, y=1, z=1}, {x=0, y=0, z=0}, rng) for bounce < 2 do
        let dists = (map (check_sphere ray) spheres) in
        let i = dists |> imin in
        if i == -1 then ({origin={x=0, y=0, z=0}, dir={x=0, y=0, z=0}}, {x=0, y=0, z=0}, {x=0, y=0, z=0}, rng)
        else
            let hit_point = ray.origin <+> (dists[i] <*> ray.dir) in
            let hit_normal = hit_point <-> spheres[i].pos |> normalize in

            -- choose random offset for diffuse reflection
            let (rng, dir) = rand_sphere_point rng in
            let dir_up = (dot dir ray.dir |> f32.sgn) <*> dir in

            ({origin=hit_point, dir=hit_normal <+> dir_up},
             {x = spheres[i].color.x * color.x,
             y = spheres[i].color.y * color.y,
             z = spheres[i].color.z * color.z},
             {x = spheres[i].emission.x + light.x * color.x,
             y = spheres[i].emission.y + light.y * color.y,
             z = spheres[i].emission.z + light.z * color.z},
             rng) in
    final_light

def trace_rays [n] rng (rays: [n]Ray) spheres =
    let rngs = randfloat.engine.split_rng n rng in
    map2 (\rng ray ->
        (map (\samplerng -> (trace_ray samplerng ray spheres)) (randfloat.engine.split_rng samples_per_ray rng)) |> vecavg |> toarr) rngs rays

def render (camera_orig: Vec, spheres: []Sphere): [height][width][3]f32 =
    let xs = (map (\x -> (2*(x+0.5) / (f32.i64 width) - 1) * f32.tan(fov/2) * (f32.i64 width)/(f32.i64 height)) (map f32.i64 (iota width))) in
    let ys = (map (\y -> (2*(y+0.5) / (f32.i64 height) - 1) * f32.tan(fov/2)) (map f32.i64 (iota height))) in
    let rays = (map (\y -> map3 (\x y z -> {origin = {x=0, y=0, z=0}, dir = normalize {x = x, y = y, z = z}}) xs (replicate width y) (replicate width (-1.0))) ys |> flatten) in
    let rng = minstd_rand.rng_from_seed [123] in
    trace_rays rng rays spheres |> unflatten height width

def main (x: f32) (y: f32) (z: f32): [height][width][3]f32 = (render({x = x, y = y, z = z}, [
    {pos = {x = -12, y = 0, z = -16}, radius = 10, color = {x=1, y=1, z=1}, emission = {x=1, y=1, z=1}},
    {pos = {x = 0, y = 0, z = -14}, radius = 1, color = {x=1, y=0, z=1}, emission = {x=0, y=0, z=0}}
    ]))
