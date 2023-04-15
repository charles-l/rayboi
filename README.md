# rayboi

A raytracer/pathtracer implemented in Futhark (and a bit of Python).

![render example](render.png)

## running the renderer
To run the renderer, build the code:

```
./futhark-nightly-linux-x86_64/bin/futhark pkg sync
./futhark-nightly-linux-x86_64/bin/futhark pyopencl rayboi.fut
```

Then run the progressive renderer shell:

```
pip install pyglet numpy
python debug_view.py
```

## other notes

A lot of the code is [based off of Sabastian Lague's very helpful video](https://www.youtube.com/watch?v=Qz0KTGYJtUk)

Future improvements/projects:

* Model loading
* Texture mapping
* Spatial partitioning with [BVH](https://raytracing.github.io/books/RayTracingTheNextWeek.html)
* [PBRT](http://www.pbr-book.org/3ed-2018/contents.html)
* ReSTIR ([ref](https://www.youtube.com/watch?v=gsZiJeaMO48))
