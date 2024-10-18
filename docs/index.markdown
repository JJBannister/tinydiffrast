---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: home
# title: About
usemathjax: true
---

Welcome to the TinyDiffRast project! 

Inspired by [tiny renderer](https://github.com/ssloy/tinyrenderer/wiki), this tiny course aims to implement, explain, and explore state-of-the-art approaches to _differentiable_ triangle rasterization.
Differentiable rasterization is a type of _inverse_ rendering approach, which allows 3D scene parameters (materials, lights, cameras, geometry) to be learned from 2D image supervision.

<center> <h2> Here are a few of the cool things that you will learn to implement <i>from scratch </i> in this course.  </h2> </center>


{% include teaser.html 
file1="/assets/images/teaser/render_small.mp4" 
file2="/assets/images/teaser/grad_small.mp4" 
file3="/assets/images/teaser/lights_small.mp4" 
file4="/assets/images/teaser/albedo_small.mp4"  
file5="/assets/images/teaser/camera_small.mp4" 
file6="/assets/images/teaser/mesh_small.mp4" 
description1="An Interactive Renderer"
description2="Jacobian/Gradient Image Generation"
description3="Inverse Rendering for Lighting"
description4="Inverse Rendering for Materials"
description5="Inverse Rendering for Cameras"
description6="Inverse Rendering for Geometry"
%}

The 3D bunny model was obtained from [The Stanford 3D Scanning Repository.](http://graphics.stanford.edu/data/3Dscanrep/)

The course content and the accompanying [codebase](https://github.com/JJBannister/tinydiffrast/tree/main/code) was developed by [Jordan J. Bannister](http://jordanbannister.ca/)
while working at [Mila](https://mila.quebec/en/), in collaboration with Derek Nowrouzezahrai from the [McGill Graphics Lab](https://www.cim.mcgill.ca/~derek/). 
Depending on what the future holds, we may develop more content like this to delve into other topics in the world of inverse rendering. 
If this is something that interests you, let us know by giving us a star on the project [github page](https://github.com/JJBannister/tinydiffrast). 


<!-- __This project is currently under development. The content may be updated or changed without warning.__ -->


<br />
# Lessons

