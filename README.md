AI Path Tracer Denoiser
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Final Project**

* Members: Dewang Sultania & Vaibhav Arcot
  - [Dewang's Linkedin](https://www.linkedin.com/in/dewang-sultania/)
  - [Vaibhav's LinkedIn](https://www.linkedin.com/in/vaibhav-arcot-129829167/)
* Tested on: Windows 10, i7-7700HQ @ 2.8GHz (3.8 Boost) 32GB, External GTX 1080Ti, 11G (personal laptop)

![Banner image](./imgs/sponza_1.gif)

## General Overview

The repo is dedicated to the implementation of the paper titled [Interactive Reconstruction of Monte Carlo Image Sequences using a Recurrent Denoising Autoencoder](https://research.nvidia.com/sites/default/files/publications/dnn_denoise_author.pdf), which proposes a purely ML approach to denoising 1 sample per pixel path tracer renders. The approach takes into account the temporal nature of the moving camera, to reduce flickering between the different frames. 

The Path tracer was implemented  path tracer written entirely in C++ and CUDA accelerated while the network was created and trained in PyTorch. Inference is done using the C++ bindings of Torch.

## Path Tracing Overview

The idea of a path tracer is to simulate the effect light and materials have on other objects in the scene. All images shown were created by running 5000 iterations unless otherwise specified, and all scene files and meshes are provided

### Features
* Reflective and diffused materials
* Stream compaction
* Material sorting
* Caching first bounce
* Refractive materials using Schlick's approximation of Fresnel's law
* Motion blur
* Anti Aliasing
* Normal debugging view
* Loading arbitrary meshes and ray culling
## Cornell Box

![Ref image](./imgs/REFERENCE_cornell.5000samp.png)

The Cornell box is a simple stage, consisting of 5 diffusive walls (1 red, 1 green and the other 3 white). In the above sample, a diffusive sphere.

### Effect of depth on a render

To show the effect of depth on the render, We decided to render a reflective intensive scene. 2 of the walls (red and green) and 6 orbs are reflective, 2 light sources (one is the middle orb), 2 transparent (green) orbs and 1 orb(blue) + 3 walls are diffusive. Because of this setup, the number of remaining rays doesn't reach 0 by a depth of 8, meaning there can be a further improvement (in deeper reflections).

| Depth | Render                   | Comment                                                      |
| ----- | ------------------------ | ------------------------------------------------------------ |
| 1     | ![](./imgs/depth/1.png)  | For this render, we see no reflections at all. The no path tracing case. |
| 2     | ![](./imgs/depth/2.png)  | We start to see some reflections (only the simplest ones).   |
| 3     | ![](./imgs/depth/3.png)  | We can see more reflections on the reflection of the orbs in the walls. |
| 4     | ![](./imgs/depth/4.png)  | We now have better refractions.                              |
| 5     | ![](./imgs/depth/5.png)  | The reflections of the orbs have some transparency.          |
| 6     | ![6](./imgs/depth/6.png) | The reflection of the transparent orbs isn't transparent.    |
| 7     | ![7](./imgs/depth/7.png) | The difference is subtle, but is shows up in the 3rd order reflections |
| 8     | ![](./imgs/depth/8.png)  | We can keep going, but here is a good stopping point.        |

 

### Effect of iterations on a render

To see the effect of iterations on render quality, we went with the same image we used above (with a depth of 8) to test the effect of iteration on render for a semi-complex scene. From visual inspection, 2000 seems to be the tipping point, and further iterations have diminishing value.

| Iterations | Render                    |
| ---------- | ------------------------- |
| 50         | ![](./imgs/iter/50.png)   |
| 250        | ![](./imgs/iter/250.png)  |
| 500        | ![](./imgs/iter/500.png)  |
| 1000       | ![](./imgs/iter/1000.png) |
| 2000       | ![](./imgs/iter/2000.png) |
| 5000       | ![](./imgs/iter/5000.png) |


## Dependencies & CMake changes

- CUDA 10+
- [tinyobjloader](https://github.com/syoyo/tinyobjloader) (Included in repo)
- Added *common.h* to the CMakeList.txt

## Building Torch Using CMake (With OpenCV)

* Download [Torch]( https://pytorch.org/tutorials/advanced/cpp_export.html ) From the official website (C++ build with or without CUDA) and extract it into the project

* Download CUDNN (Required for Torch) and extract into the project

* First, add the necessary lines from [this post]( https://pytorch.org/tutorials/advanced/cpp_export.html ) to your CMakeLists.txt file (Already one for this project)

* In the build directory, run ```cmake-gui ..``` and point cmake towards the OpenCV build directory (if not already in the path)

* Then close cmake-gui (This is the order because OpenCV is found before Torch) and run the following command (Point it towards the libtorch relative to the build directory)

  ```cmake -DCMAKE_PREFIX_PATH=$PWD/../../libtorch ..```

* Run ```cmake-gui ..``` again and now point ```CUDNN_INCLUDE_PATH``` towards the include folder inside the CUDNN folder.

* Point ```CUDNN_LIBRARY_PATH``` to the library file (```/absolute/path/to/CUDNN/lib/x64/cudnn.lib```)

* Generate the project

## Useful links

[3D obj files with normals](https://casual-effects.com/data/)

[Fresnel's law](https://blog.demofox.org/2017/01/09/raytracing-reflection-refraction-fresnel-total-internal-reflection-and-beers-law/)

[Easier 3D obj files](https://graphics.cmlab.csie.ntu.edu.tw/~robin/courses/cg04/model/index.html)