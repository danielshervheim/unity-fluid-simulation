# Fluid Simulation

![image](https://i.imgur.com/naG9ADa.jpg)

See it in action [here](http://www.youtube.com/watch?v=aUgFWNUzMw0).

A Eularian fluid simulation  based on Jos Stam's paper *[Real-Time Fluid Dynamics for Games](https://pdfs.semanticscholar.org/847f/819a4ea14bd789aca8bc88e85e906cfc657c.pdf)*, implemented in [Unity](https://unity3d.com/).

In CPU based fluid solvers, the bottleneck is generally the size of the simulation grid. To alleviate these problems, I implemented my solver on the GPU via Unity's ComputeShader interface. This greatly increased the speed of the resulting simulations. My computer was able to simulate a 1024 × 1024 grid at interactive framerates (> 60 fps).

I also included a CPU-only implementation for reference. (In comparison, my computer was only able to simulate a 64 × 64 grid at interactive framerates with the CPU version).
