---
layout: post
title:  "12) Wrapping Up"
author: "Jordan J. Bannister"
---

Welcome back! 
Over the past 12 lessons, we have discussed and implemented _from scratch_ a differentiable rasterization pipeline including two different state-of-the-art algorithms for solving the discontinuity problem.
We have also used the pipeline to perform inverse rendering of lights, cameras, materials, and geometry.
This lesson will mostly be a high level discussion of differentiable rasterization, and some suggestions and ideas for where you could go next on your journey in the world of inverse rendering.


__Contents__
* This will become a table of contents (this text will be scrapped).
{:toc}

# Strengths and Weaknesses of Differentiable Triangle Rasterization

We believe that differentiable triangle rasterization is an excellent way to get your feet wet in the domain of inverse rendering.
That's why we made this course.
However, that doesn't mean that it is the best choice for every application.
Therefore, if you are considering using differentiable rasterization in a project, you should consider the pros and cons of a differentiable rasterization approach.


## Weaknesses
### Realistic Camera Models
One of the defining features of a rasterization approach is the process of mapping triangles (or any geometry) from world space to screen space.
This decision is central to the speed and efficiency of rasterization.
This approach works very well for pinhole and orthographic cameras.
But, how often do you see a pinhole or an orthographic camera in real life? 
If you are familiar with photography, or have viewed a partial eclipse, you may have made a camera with an actual pinhole at some point.
This is a cool science experiment, but is not generally a tool that photographers use...
Furthermore, this type of pinhole camera is different from what we modelled in our codebase.
Our virtual pinhole camera has an infinitely small aperature. 
Even if you could make an infinitely small pinhole in real life, it would not collect any light!
You may also be familiar with [telecentric lenses](https://www.opto-e.com/en/resources/tutorials/telecentric-lenses-tutorial).
However, these lenses are very rare outside of niche industrial applications.

In the real world, most cameras are made with lens systems that have some ammount of distortion and have non-pinhole aperatures.
Cameras that have non-pinhole aperatures produce effects like [depth of field](https://en.wikipedia.org/wiki/Depth_of_field) and [bokeh](https://en.wikipedia.org/wiki/Bokeh).
If you want to capture these effects faithfully in a renderer, you may need to move beyond rasterization. 
The same is true for differentiable rendering.

### Shadows and Indirect Lighting

The rasterization approach, in which geometry is transformed to screen space, is very helpful for detecting primary ray intersections. 
This means detecting intersections between scene geometry and ray paths associated directly with pixels in a camera model.
However, there are many optical effects that are not related to primary intersections.
One example of this is the phenomenon of [shadows](https://en.wikipedia.org/wiki/Shadow).
Shadows are the result of geometry blocking or re-directing the path of light elsewhere in a scene.
Intersections of that sort can also reflect or refract light so as to illuminate other geometry with light that did not directly come from a light source.
This phenomenon is called [indirect illumination](https://blogs.nvidia.com/blog/direct-indirect-lighting/).

Unfortunately, the process of transforming geometry to screen space is not much help at all once we begin dealing with non-primary ray intersections.
In the real world, light tends to scatter all over in a non-coherent way.
There are some techniques to appoximate effects like shadows in rasterization, but they are approximations.
In general, if you want to capture these effects faithfully in a render, you need to move beyond rasterization. 
The same is true for differentiable rendering.

### Triangle Meshes

The approach of representing geometry using triangle meshes, is a very useful and efficient way of representing surface geometry.
Triangles can be easily mapped to screen space, and the half-plane algorithm works very well for drawing triangles on a screen.
However, in the real world, not all geometry is well represented as a surface.
Consider a cloud, or fog, or steam. 
How could we represent those effectively with triangles?

We also saw in our experiments with optimizing mesh geometry that triangle meshes can become quite messy during optimization.
This is emblematic of a more general problem related to changes in topology during optimization.
Imagine if, instead of transforming our sphere into a bunny, we wished to transform the sphere into a doughnut.
To properly handle this deformation, we would need to break and form triangle connections in our mesh to create the doughnut hole.
We may also wish to delete and create triangles in our mesh, depending on the optimization task.
This is a very tricky thing to do well. 
There has been some work on how to handle [dynamic mesh topology](https://research.nvidia.com/labs/toronto-ai/DMTet/assets/dmtet.pdf), but it relies on an implicit surface representation as well as a triangle mesh.

In general, if you want to capture volumetric geometry, or topological changes in geometry, you may need to use representations other than triangle meshes.

## Strengths
All of this being said, differentiable triangle rasterization has several notable strengths.
For one, it is very efficient in terms of compute and memory!
This is not something to be underestimated in a domain where computational costs can easily become prohibitive.
Many of the methods to address the shortcomings of triangle rasterization require significantly more compute and/or memory.
Therefore, if differentiable triangle rasterization allows you to solve your problem, it is likely a good choice.

Another notable advantage is that the use of triangle mesh geometry and triangle rasterization for rendering are both ubiquitous across digital 3D industries.
These techniques and data formats are still around today after decades of use and will almost certainly be around for many years to come.
They are deeply integrated into many existing software products and tech stacks.
Differentiable rendering approaches that don't require a new geometric representation or rendering algorithm may be easier to integrate into existing systems and workflows.


# Adjacent Work
Outside of differentiable rasterization, there are several other threads in the space of inverse rendering that are worth knowing about and keeping an eye on.

## Ray Tracing
If your goal is to create the most realistic renderings, the technique of ray tracing is probably the direction that you want to pursue.
Just make sure you have access to some nice GPUs!
Compared to rasterization, ray tracing is much more computationally expensive. 
It can also simulate many effects that rasterization cannot. 
This is because it models the way in which light can undergo many different interactions before being picked up by a camera.
This also means that differentiable ray tracing techniques are able to optimize for scene parameters and account for optical phenomenon in ways that rasterization is not able to.

Just as there are differentiable rasterization techniques to solve the discontinuity problem, there are differentiable ray tracing techniques to solve the same problem.
The state of the art in this space involves integral re-parameterization ([1](http://people.csail.mit.edu/sbangaru/projects/was-2020/index.html), [2](https://shuangz.com/projects/psdr-was-sa23/)). 
The most advanced differentiable ray tracing framework currently is [Mitsuba3](https://www.mitsuba-renderer.org/) and the lab leading the developent of Mitsuba3 publishes many [articles](https://rgl.epfl.ch/people/wjakob) on differentiable ray tracing (so it is a great place to start reading).

## Radiance Fields

In a certain sense, radiance fields are a step in the opposite direction compared to ray tracing.
Radiance field rendering doesn't bother simulating any light bounces at all (even the direct illumination that we implemented in our rasterization pipeline).
Instead of having light sources that illuminate non-emmisive geometric objects, radiance fields treat all geometry as emmisive.
In other words, radiance fields capture a distribution of [radiance](https://en.wikipedia.org/wiki/Radiance) (directionally emmitted light) across a 3D scene.
During rendering, radiance is accumulated along the primary rays that are associated with camera pixels.
Initially, neural networks were used to represent radiance distributions and this technique was dubbed neural radiance fields (or NeRFs).
Currently the state-of-the-art in this space uses [Gaussian splats](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) instead of neural networks.

The advantage of this approach is that neural networks and Gaussian splats are very flexible representations. 
They can learn complex and realistic scenes from a series of images or a video.
They can also capture volumetric (e.g. clouds, mist) and semi-volumetric (e.g. fur) geometry quite well. 
The main downside of this approach is that it don't model light transport in an accurate way.
For example, in a rasterization approach, if a geometry element is altered (e.g. rotating a bunny mesh), the appearance of the bunny will change in a realistic way.
Some areas of the bunny will change color because the relation of the bunny to the lighting enviroment changes and the shader output is different as a result.
In other words, the rasterization approach is actually simulating light transport to some degree.
In a radiance field approach, rotating an object will not result in the appearance of the object to be updated in an accurate way.
This is because the appearance of each object is not derived from a physically accurate light transport simulation.
Shadows, specular highlights, and even reflections are "baked" into the scene rather than arising from a light transport simulation.

This is a very active are of research and there has been some recent [work](https://lzhnb.github.io/project-pages/gs-ir.html) on combining radiance fields and other rendering techniques to achieve the best of both worlds (so to speak).
I expect that we will see some very impressive advancements in this space over the coming years.

# Final Coding Challenges

Before, we conclude, we have compiled a list of additional coding challenges for you!
Just in case you haven't had enough already.
You won't find solutions to these challenges in our codebase, so you will be on your own from here.
However, if you've made it this far, we're confident that you have the skills to tackle them.

### Easy/Moderate
- Extend the camera classes so that you can also learn field-of-view (perspective) or projection size (orthographic).
- Implement a clipping stage in the Rasterizer.
- Improve the rasterizer to run faster on GPU, CPU or on a specific scene.
- Implement an experiment that optimizes material properties other than albedo (like the specular coefficient).
- Implement a different shading model.
- Implement UV mapping and texture lookup so that material properties can be defined using images.
- Implement a different lighting model.
- Implement a different mesh regularization approach.
- Implement a gradient descent algorithm that incorporates momentum.

### Hard
- Implement a new or improved approach to solve the discontinuity problem.
- Implement support for semi-transparent meshes.
- Implement a [differentiable shadow mapping](https://openaccess.thecvf.com/content/CVPR2023/papers/Worchel_Differentiable_Shadow_Mapping_for_Efficient_Inverse_Graphics_CVPR_2023_paper.pdf) approach.
- Implement a method that supports [dynamic mesh topology](https://research.nvidia.com/labs/toronto-ai/DMTet/assets/dmtet.pdf).
- Implement a method to approximate [depth of field](https://dl.acm.org/doi/abs/10.1145/3550454.3555521).


# Conclusion
If you have stuck with us throught this entire course, and especially if you completed all of the coding challenges, then congratulations!
We hope that you found the course valuable and are very grateful that you took the time to engage with our work.