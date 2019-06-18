# Fluid Simulation

![image](https://i.imgur.com/naG9ADa.jpg)

Watch on [Youtube](http://www.youtube.com/watch?v=aUgFWNUzMw0).

## About

A fluid simulation I made for a class assignment, based on Jos Stam's excellent paper *[Real-Time Fluid Dynamics for Games](https://pdfs.semanticscholar.org/847f/819a4ea14bd789aca8bc88e85e906cfc657c.pdf)*.



In fluid solvers, the bottleneck is generally the size of the simulation grid. To improve upon Stam's original work, I implemented my solver on the GPU via Compute shaders. (I also included a CPU version for reference).



The GPU version can handle a **1024 x 1024** grid at 60 fps, while the CPU version can only handle about a **64 x 64** grid.  (As tested on a Ryzen 5, Nvidia GTX 1060 machine. Your mileage may vary).



## To Use

Right-click to deposit ink.

Left-click to swirl the liquid around.

Middle-click to reset.