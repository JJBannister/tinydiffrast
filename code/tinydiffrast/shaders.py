from typing import List

import random
import taichi as ti
import taichi.math as tm

from .rasterizer import TriangleRasterizer
from .lights import DirectionalLightArray
from .cameras import Camera


@ti.data_oriented
class BlinnPhongShader:

    def __init__(
        self,
        camera: Camera,
        directional_light_array: DirectionalLightArray,
        triangle_id_buffer: ti.field,
        world_position_buffer: ti.Vector.field,
        normals_buffer: ti.Vector.field,
        albedo_buffer: ti.Vector.field,
        ambient_constant: float = 0.1,
        specular_constant: float = 0.1,
        diffuse_constant: float = 0.1,
        shine_coefficient: float = 10,
    ) -> None:

        self.camera = camera
        self.light_array = directional_light_array

        self.triangle_id_buffer = triangle_id_buffer
        self.world_position_buffer = world_position_buffer
        self.normals_buffer = normals_buffer
        self.albedo_buffer = albedo_buffer

        self.ambient_constant = ti.field(float, shape=())
        self.specular_constant = ti.field(float, shape=())
        self.diffuse_constant = ti.field(float, shape=())
        self.shine_coefficient = ti.field(float, shape=())

        self.ambient_constant[None] = ambient_constant
        self.specular_constant[None] = specular_constant
        self.diffuse_constant[None] = diffuse_constant
        self.shine_coefficient[None] = shine_coefficient

        self.output_buffer = ti.Vector.field(
            n=3, dtype=float, shape=albedo_buffer.shape, needs_grad=True
        )

    def forward(self):
        self.clear()
        self.shade()

    def clear(self):
        self.output_buffer.fill(0)
        self.output_buffer.grad.fill(0)

    def backward(self):
        self.shade.grad()

    @ti.kernel
    def shade(self):
        for x, y, l in self.output_buffer:
            if self.triangle_id_buffer[x, y, l] != 0:

                # Grab values required for shading
                albedo = self.albedo_buffer[x, y, l]
                normal_vector = tm.normalize(self.normals_buffer[x, y, l])
                world_position = self.world_position_buffer[x, y, l]
                camera_position = self.camera.rigid_transform.position_param[None]
                view_vector = tm.normalize(camera_position - world_position)

                # Ambient component
                output_color = albedo * self.ambient_constant[None]

                # Add specular and diffuse component per light
                for i in ti.static(range(self.light_array.n_lights)):
                    light_vector = -tm.normalize(self.light_array.light_directions[i])
                    light_color = self.light_array.light_colors[i]

                    # Specular
                    halfway_vector = tm.normalize(light_vector + view_vector)
                    specular_magnitude = tm.pow(
                        tm.max(0, tm.dot(normal_vector, halfway_vector)),
                        self.shine_coefficient[None],
                    )
                    output_color += (
                        self.specular_constant[None] * specular_magnitude * light_color
                    )

                    # Diffuse
                    diffuse_magnitude = tm.max(0, tm.dot(light_vector, normal_vector))
                    output_color += (
                        self.diffuse_constant[None] * diffuse_magnitude * light_color * albedo
                    )

                self.output_buffer[x, y, l] = output_color


@ti.data_oriented
class SilhouetteShader:

    def __init__(
        self,
        triangle_id_buffer: ti.Vector.field,
        triangle_groups: ti.field = None,
    ) -> None:

        self.output_buffer = ti.Vector.field(
            n=3, dtype=float, shape=triangle_id_buffer.shape
        )

        self.triangle_id_buffer = triangle_id_buffer

        self.use_group_colors = triangle_groups is not None
        self.triangle_groups = triangle_groups

        self.color_0 = 0.5,0.5,0.5
        self.color_1 = 1,1,1

    def forward(self):
        self.shade()

    @ti.kernel
    def shade(self):
        for i, j, k in self.output_buffer:
            tri_id = self.triangle_id_buffer[i, j, k]

            if tri_id == 0:
                self.output_buffer[i, j, k] = 0

            elif ti.static(self.use_group_colors):
                group_id = self.triangle_groups[tri_id - 1] 
                if group_id == 0:
                    self.output_buffer[i, j, k] = self.color_0
                if group_id == 1:
                    self.output_buffer[i, j, k] = self.color_1

            else:
                self.output_buffer[i, j, k] = 1
