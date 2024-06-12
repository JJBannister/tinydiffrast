import taichi as ti
import taichi.math as tm

ti.init(arch=ti.cuda)

import tinydiffrast as tdr


def main():
    pipe = Pipeline(resolution=(1024, 1024), mesh_type="bunny")

    # Set true vertex albedos
    pipe.vertex_albedos.fill(tm.vec3([0.9, 0.1, 0.9]))

    # Get reference image and set as reference
    pipe.forward()
    true_image = pipe.output_buffer.to_numpy()
    pipe.mse.set_reference_image(true_image)

    # Perturb before training
    pipe.vertex_albedos.fill(tm.vec3([0.1, 0.9, 0.1]))

    # Get initial image and show plots
    pipe.forward()
    initial_image = pipe.output_buffer.to_numpy()

    tdr.plot_image(true_image, title="True Image")
    tdr.plot_image(initial_image, title="Initial Image")
    tdr.show_plots()

    # Training
    window = ti.ui.Window("Live Training Progress", res=pipe.resolution)
    canvas = window.get_canvas()
    canvas.set_image(pipe.output_buffer)
    window.show()

    i = 0
    while window.running:
        i += 1
        pipe.forward()
        canvas.set_image(pipe.output_buffer)
        window.show()

        if i % 100 == 0:
            loss = pipe.mse.loss[None]
            print("Train step: {}, loss: {}".format(i, loss))

        pipe.backward()
        pipe.update()

    # Post-Train Visualize
    final_image = pipe.output_buffer.to_numpy()
    tdr.plot_image(true_image, title="True Image")
    tdr.plot_image(initial_image, title="Initial Image")
    tdr.plot_image(final_image, title="Final Image")
    tdr.show_plots()


@ti.data_oriented
class Pipeline:

    def __init__(
        self,
        resolution=(1024, 1024),
        mesh_type="dragon",
    ):
        self.mesh = tdr.TriangleMesh.get_example_mesh(mesh_type)
        self.resolution = resolution
        self.lr = 5000

        self.camera = tdr.PerspectiveCamera()

        self.rasterizer = tdr.TriangleRasterizer(
            resolution, self.camera, self.mesh, subpixel_bits=8
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
        )

        self.vertex_albedos = ti.Vector.field(
            n=3, dtype=float, shape=(self.mesh.n_vertices), needs_grad=True
        )
        self.albedo_interpolator = tdr.TriangleAttributeInterpolator(
            vertex_attributes=self.vertex_albedos,
            triangle_vertex_ids=self.mesh.triangle_vertex_ids,
            triangle_id_buffer=self.rasterizer.triangle_id_buffer,
            barycentric_buffer=self.rasterizer.barycentric_buffer,
        )

        self.light_array = tdr.DirectionalLightArray(n_lights=2)
        self.light_array.set_light_color(0, tm.vec3([1, 1, 1]))
        self.light_array.set_light_color(1, tm.vec3([1, 1, 1]))
        self.light_array.set_light_direction(0, tm.vec3([-1, 0, -1]))
        self.light_array.set_light_direction(1, tm.vec3([1, 0, -1]))

        self.phong_shader = tdr.BlinnPhongShader(
            camera=self.camera,
            directional_light_array=self.light_array,
            triangle_id_buffer=self.rasterizer.triangle_id_buffer,
            world_position_buffer=self.world_position_interpolator.attribute_buffer,
            normals_buffer=self.normals_interpolator.attribute_buffer,
            albedo_buffer=self.albedo_interpolator.attribute_buffer,
            ambient_constant=0.1,
            specular_constant=0.2,
            diffuse_constant=0.3,
            shine_coefficient=30,
        )

        self.output_buffer = ti.Vector.field(
            n=3,
            dtype=float,
            shape=(self.rasterizer.x_resolution, self.rasterizer.y_resolution),
            needs_grad=True,
        )
        self.mse = tdr.ImageMeanSquaredError(self.output_buffer)

        # We are only learning albedo in this example, so we only need to run these kernels once.
        self.camera.forward()
        self.rasterizer.forward()
        self.world_position_interpolator.forward()
        self.normals_estimator.forward()
        self.normals_interpolator.forward()

    def forward(self):
        self.albedo_interpolator.forward()
        self.phong_shader.forward()
        self.copy_output_buffer()
        self.mse.forward()

    def clear(self):
        self.output_buffer.fill(0)
        self.output_buffer.grad.fill(0)

    @ti.kernel
    def copy_output_buffer(self):
        for x, y in self.output_buffer:
            self.output_buffer[x, y] = self.phong_shader.output_buffer[x, y, 0]

    def backward(self):
        self.mse.backward()
        self.copy_output_buffer.grad()
        self.phong_shader.backward()
        self.albedo_interpolator.backward()

    @ti.kernel
    def update(self):
        for i in self.vertex_albedos:
            self.vertex_albedos[i] -= self.lr * self.vertex_albedos.grad[i]
        self.vertex_albedos.grad.fill(0)


if __name__ == "__main__":
    main()
