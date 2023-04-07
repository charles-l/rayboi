def width: i64 = 800
def height: i64 = 600
def fov: f32 = f32.pi / 3

type Vec = {x: f32, y: f32, z: f32}
type Ray = {origin: Vec, dir: Vec}
type Sphere = { pos: Vec, radius: f32, color: Vec }

def toarr (a: Vec) = [a.x, a.y, a.z]
def f32toarr(f: f32) = [f, f, f]

def (a: Vec) <+> (b: Vec): Vec = {x = a.x + b.x, y = a.y + b.y, z = a.z + b.z}
def (a: Vec) <-> (b: Vec): Vec = {x = a.x - b.x, y = a.y - b.y, z = a.z - b.z}
def (s: f32) <*> (b: Vec): Vec = {x = s * b.x, y = s * b.y, z = s * b.z}
def dot (a: Vec) (b: Vec): f32 = (a.x*b.x) + (a.y*b.y) + (a.z*b.z)
def normalize(a: Vec): Vec = (1 / (dot a a |> f32.sqrt)) <*> a

def imin [n] (arr: [n]f32): i64 =
    let (i, _) = reduce (\(ai, ax) (i, x) -> (if x < ax then (i, x) else (ai, ax))) (-1, f32.inf) (zip (iota n) arr) in
    i



def check_sphere (ray: Ray) ({pos, radius, color}: Sphere): f32 =
    let ray_origin = ray.origin <-> pos in
    let s = f32.sqrt(((dot ray_origin ray.dir) * (dot ray_origin ray.dir)) - ((dot ray_origin ray_origin) - radius*radius)) in
    let d1 = (dot (-1 <*> ray_origin) ray.dir) + s in
    let d2 = (dot (-1 <*> ray_origin) ray.dir) - s in
    let d = f32.min d1 d2 in
    if f32.isnan d || d < 0 then f32.inf else d

def trace_ray (ray: Ray) (spheres: []Sphere): Vec =
    let i = (imin (map (check_sphere ray) spheres)) in
    if i == -1 then {x=0, y=0, z=0} else spheres[i].color

def render (camera_orig: Vec, spheres: []Sphere): [height][width][3]f32 =
    let xs = (map (\x -> (2*(x+0.5) / (f32.i64 width) - 1) * f32.tan(fov/2) * (f32.i64 width)/(f32.i64 height)) (map f32.i64 (iota width))) in
    let ys = (map (\y -> (2*(y+0.5) / (f32.i64 height) - 1) * f32.tan(fov/2)) (map f32.i64 (iota height))) in
    let rays = (map (\y -> map3 (\x y z -> {origin = {x=0, y=0, z=0}, dir = normalize {x = x, y = y, z = z}}) xs (replicate width y) (replicate width (-1.0))) ys |> flatten) in

    unflatten height width (map (\r -> (trace_ray r spheres) |> toarr) rays)

def main (x: f32) (y: f32) (z: f32): [height][width][3]f32 = (render({x = x, y = y, z = z}, [
    {pos = {x = -2, y = 0, z = -16}, radius = 2, color = {x=1, y=1, z=1}},
    {pos = {x = 0, y = 0, z = -20}, radius = 1, color = {x=1, y=0, z=1}}
    ]))
