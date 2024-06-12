import taichi as ti
import taichi.math as tm
import numpy as np
from random import random

ti.init(arch=ti.cuda, random_seed=1)

import tinydiffrast as tdr


def main():
    x_resolution = 128
    y_resolution = 128

    pipeline = Pipeline(
        resolution=(x_resolution, y_resolution),
        mesh_type="sphere",
    )
    pipeline.camera.set_view_parameters(
        position=(0, 0, 3.5), focal_point=(0, 0, 0), up_vector=(0, 1, 0)
    )

    pipeline.forward()
    img = pipeline.output_buffer.to_numpy()

    tdr.plot_image(img[:, :, 0, :], title="Image")
    tdr.save_image(img[:, :, 0, :], "./pics/teaser/image.png")
    tdr.show_plots()

    # Light grads
    x_grad_image = np.zeros(shape=(x_resolution, y_resolution))
    y_grad_image = np.zeros(shape=(x_resolution, y_resolution))

    for x in range(x_resolution):
        print("Processing column ", x)
        for y in range(y_resolution):
            pipeline.forward()
            pipeline.output_buffer.grad[x, y, 0] = tm.vec3([1.0, 1.0, 1.0])
            pipeline.backward()

            x_grad_image[x, y] = pipeline.light_array.light_directions.grad[0].x
            y_grad_image[x, y] = pipeline.light_array.light_directions.grad[0].y

            pipeline.light_array.light_directions.grad.fill(0)

    tdr.plot_image(grad_to_color(x_grad_image), title="X grads")
    tdr.plot_image(grad_to_color(y_grad_image), title="Y grads")
    tdr.show_plots()


@ti.data_oriented
class Pipeline:

    def __init__(
        self,
        resolution=(1024, 1024),
        mesh_type="dragon",
    ):

        self.camera = tdr.PerspectiveCamera()
        self.mesh = tdr.TriangleMesh.get_example_mesh(mesh_type)

        self.rasterizer = tdr.TriangleRasterizer(
            resolution, self.camera, self.mesh, subpixel_bits=4
        )
        self.world_position_interpolator = tdr.TriangleAttributeInterpolator(
            vertex_attributes=self.mesh.vertices,
            triangle_vertex_ids=self.mesh.triangle_vertex_ids,
            triangle_id_buffer=self.rasterizer.triangle_id_buffer,
            barycentric_buffer=self.rasterizer.barycentric_buffer,
        )

        self.normals_estimator = tdr.NormalsEstimator(mesh=self.mesh)
        self.normals_interpolator = tdr.TriangleAttributeInterpolator(
            vertex_attributes=self.normals_estimator.normals,
            triangle_vertex_ids=self.mesh.triangle_vertex_ids,
            triangle_id_buffer=self.rasterizer.triangle_id_buffer,
            barycentric_buffer=self.rasterizer.barycentric_buffer,
            normalize=True,
        )

        self.albedo_shader = tdr.SilhouetteShader(
            triangle_id_buffer=self.rasterizer.triangle_id_buffer
        )

        self.light_array = tdr.DirectionalLightArray(n_lights=1)
        self.light_array.set_light_color(0, tm.vec3([1, 1, 1]))
        self.light_array.set_light_direction(0, tm.vec3([1, -1, -1]))

        self.phong_shader = tdr.BlinnPhongShader(
            camera=self.camera,
            directional_light_array=self.light_array,
            triangle_id_buffer=self.rasterizer.triangle_id_buffer,
            world_position_buffer=self.world_position_interpolator.attribute_buffer,
            normals_buffer=self.normals_interpolator.attribute_buffer,
            albedo_buffer=self.albedo_shader.output_buffer,
            ambient_constant=0.1,
            specular_constant=0.2,
            diffuse_constant=0.3,
            shine_coefficient=30,
        )

        self.output_buffer = self.phong_shader.output_buffer

    def forward(self):
        self.camera.forward()
        self.rasterizer.forward()
        self.world_position_interpolator.forward()
        self.normals_estimator.forward()
        self.normals_interpolator.forward()
        self.albedo_shader.forward()
        self.phong_shader.forward()

    def backward(self):
        self.phong_shader.backward()


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
