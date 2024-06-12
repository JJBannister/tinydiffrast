from enum import Enum

import taichi as ti
import taichi.math as tm
import numpy as np
from random import random

ti.init(arch=ti.cuda)


import tinydiffrast as tdr


def main():
    x_resolution = 128
    y_resolution = 128

    pipeline = Pipeline(
        resolution=(x_resolution, y_resolution),
        mesh_type="sphere_background",
        discontinuity_mode=Pipeline.DiscontinuityMode.RTS,
        single_layer_mode=False,
        translation_vertex_group=0,
    )

    pipeline.forward()
    img = pipeline.output_buffer.to_numpy()

    tdr.plot_image(img[:, :, :], title="Image")
    tdr.show_plots()

    # Translation grads
    x_grad_image = np.zeros(shape=(x_resolution, y_resolution))
    y_grad_image = np.zeros(shape=(x_resolution, y_resolution))

    for x in range(x_resolution):
        print("Processing column ", x)
        for y in range(y_resolution):
            pipeline.forward()
            pipeline.output_buffer.grad[x, y] = tm.vec3([1.0, 1.0, 1.0])
            pipeline.backward()

            x_grad_image[x, y] = (
                pipeline.rigid_mesh_transform.rigid_transform.position_param.grad[
                    None
                ].x
            )
            y_grad_image[x, y] = (
                pipeline.rigid_mesh_transform.rigid_transform.position_param.grad[
                    None
                ].y
            )

    tdr.plot_image(grad_to_color(x_grad_image), title="X grads")
    tdr.plot_image(grad_to_color(y_grad_image), title="Y grads")
    tdr.show_plots()


@ti.data_oriented
class Pipeline:

    class DiscontinuityMode(Enum):
        RTS = 1
        NVDR = 2
        NONE = 3

    def __init__(
        self,
        resolution=(1024, 1024),
        mesh_type="dragon",
        discontinuity_mode=DiscontinuityMode.NONE,
        single_layer_mode=False,
        translation_vertex_group=0,
    ):
        self.camera = tdr.PerspectiveCamera()
        self.camera.set_view_parameters(
            position=(0, 0, 3), focal_point=(0, 0, 0), up_vector=(0, 1, 0)
        )

        self.mesh = tdr.TriangleMesh.get_example_mesh(mesh_type)
        self.rigid_mesh_transform = tdr.RigidMeshTransform(
            mesh=self.mesh, vertex_group=translation_vertex_group
        )
        self.transformed_mesh = self.rigid_mesh_transform.transformed_mesh

        self.rasterizer = tdr.TriangleRasterizer(
            resolution, self.camera, self.transformed_mesh, subpixel_bits=4
        )
        self.silhouette_shader = tdr.SilhouetteShader(
            triangle_id_buffer=self.rasterizer.triangle_id_buffer,
            triangle_groups=self.transformed_mesh.triangle_groups,
        )

        # Different paths depending on different modes
        # NVDR vs. RTS vs. Nothing
        self.discontinuity_mode = discontinuity_mode

        if discontinuity_mode == self.DiscontinuityMode.NONE:
            self.output_buffer = ti.Vector.field(
                n=3,
                dtype=float,
                shape=(self.rasterizer.x_resolution, self.rasterizer.y_resolution),
                needs_grad=True,
            )

        elif discontinuity_mode == self.DiscontinuityMode.RTS:
            self.world_space_interpolator = tdr.TriangleAttributeInterpolator(
                vertex_attributes=self.transformed_mesh.vertices,
                triangle_vertex_ids=self.transformed_mesh.triangle_vertex_ids,
                triangle_id_buffer=self.rasterizer.triangle_id_buffer,
                barycentric_buffer=self.rasterizer.barycentric_buffer,
            )
            self.splat_position_interpolator = tdr.TriangleSplatPositionInterpolator(
                camera=self.camera,
                world_space_buffer=self.world_space_interpolator.attribute_buffer,
                triangle_id_buffer=self.rasterizer.triangle_id_buffer,
            )
            self.triangle_splatter = tdr.TriangleSplatter(
                shade_buffer=self.silhouette_shader.output_buffer,
                screen_space_buffer=self.splat_position_interpolator.screen_space_buffer,
                depth_buffer=self.rasterizer.depth_buffer,
                triangle_id_buffer=self.rasterizer.triangle_id_buffer,
                single_layer_mode=single_layer_mode,
            )

            self.output_buffer = self.triangle_splatter.output_buffer

        elif discontinuity_mode == self.DiscontinuityMode.NVDR:
            self.antialiaser = tdr.Antialiaser(
                shade_buffer=self.silhouette_shader.output_buffer,
                depth_buffer=self.rasterizer.depth_buffer,
                triangle_id_buffer=self.rasterizer.triangle_id_buffer,
                clip_vertices=self.rasterizer.clip_vertices,
                triangle_vertex_ids=self.transformed_mesh.triangle_vertex_ids,
            )

            self.output_buffer = self.antialiaser.output_buffer

        else:
            raise Exception("discontinuity_mode must be one of [RTS, NVDR, NONE]")

    @ti.kernel
    def copy_output_buffer(self):
        for x, y in self.output_buffer:
            self.output_buffer[x, y] = self.silhouette_shader.output_buffer[x, y, 0]

    def forward(self):
        self.camera.forward()
        self.rigid_mesh_transform.forward()
        self.rasterizer.forward()
        self.silhouette_shader.forward()

        if self.discontinuity_mode == self.DiscontinuityMode.NONE:
            self.copy_output_buffer()

        elif self.discontinuity_mode == self.DiscontinuityMode.RTS:
            self.world_space_interpolator.forward()
            self.splat_position_interpolator.forward()
            self.triangle_splatter.forward()

        elif self.discontinuity_mode == self.DiscontinuityMode.NVDR:
            self.antialiaser.forward()

    def backward(self):
        if self.discontinuity_mode == self.DiscontinuityMode.NONE:
            self.copy_output_buffer.grad()
            self.rasterizer.backward()

        elif self.discontinuity_mode == self.DiscontinuityMode.RTS:
            self.triangle_splatter.backward()
            self.splat_position_interpolator.backward()
            self.world_space_interpolator.backward()

        elif self.discontinuity_mode == self.DiscontinuityMode.NVDR:
            self.antialiaser.backward()
            self.rasterizer.backward()

        self.rigid_mesh_transform.backward()


def grad_to_color(img: np.array) -> np.array:
    mag_img = np.absolute(img)
    mag_img = mag_img / np.max(mag_img)

    red_img = np.where(img > 0, mag_img, np.zeros(shape=img.shape))
    blue_img = np.where(img < 0, mag_img, np.zeros(shape=img.shape))

    color_img = np.zeros(shape=(img.shape[0], img.shape[1], 3))
    color_img[:, :, 0] = red_img
    color_img[:, :, 2] = blue_img
    return color_img


if __name__ == "__main__":
    main()
