from enum import Enum

import taichi as ti
import taichi.math as tm

ti.init(arch=ti.cuda)

import tinydiffrast as tdr


def main():
    pipe = Pipeline(
        resolution=(1024, 1024),
        mesh="dragon",
        discontinuity_mode=Pipeline.DiscontinuityMode.RTS,
    )

    # Create ground truth image
    pipe.camera.set_view_parameters(
        position=(2, 1, 4),
        focal_point=(0, 0, 0),
        up_vector=(0.5, 1, 0),
    )
    pipe.forward()
    true_image = pipe.output_buffer.to_numpy()
    pipe.mse.set_reference_image(true_image)

    # Perturb camera parameters before training
    pipe.camera.set_view_parameters(
        position=(1, 2, 4),
        focal_point=(0.5, 0.5, 0),
        up_vector=(-0.1, 1, 0),
    )
    pipe.forward()
    initial_image = pipe.output_buffer.to_numpy()

    # Pre-Train Visualize
    tdr.plot_image(true_image, title="True Image")
    tdr.plot_image(initial_image, title="Initial Image")

    initial_delta_img = 0.5 + (initial_image - true_image) / 2.0
    tdr.plot_image(initial_delta_img, title="Initial Delta Image")
    tdr.show_plots()

    # Train
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

        pipe.backward()
        pipe.update()

        if i % 100 == 0:
            loss = pipe.mse.loss[None]
            print("Train step: {}, loss: {}".format(i, loss))

    # Post-Train Visualize
    final_image = pipe.output_buffer.to_numpy()
    final_delta_img = 0.5 + (final_image - true_image) / 2.0
    tdr.plot_image(final_delta_img, title="Delta image after {} steps".format(i))
    tdr.show_plots()


class Pipeline:

    class DiscontinuityMode(Enum):
        RTS = 1
        NVDR = 2

    def __init__(
        self,
        resolution=(1024, 1024),
        mesh="dragon",
        discontinuity_mode: DiscontinuityMode = DiscontinuityMode.RTS,
    ):

        self.discontinuity_mode = discontinuity_mode

        self.resolution = resolution
        self.lr_position = 5e-2
        self.lr_rotation = 5e-3

        self.mesh = tdr.TriangleMesh.get_example_mesh(mesh)
        self.camera = tdr.PerspectiveCamera()
        self.rasterizer = tdr.TriangleRasterizer(
            resolution, self.camera, self.mesh, subpixel_bits=8
        )

        self.silhouette_shader = tdr.SilhouetteShader(
            self.rasterizer.triangle_id_buffer, self.mesh.triangle_groups
        )

        if discontinuity_mode == self.DiscontinuityMode.RTS:
            self.world_space_interpolator = tdr.TriangleAttributeInterpolator(
                vertex_attributes=self.mesh.vertices,
                triangle_vertex_ids=self.mesh.triangle_vertex_ids,
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
                single_layer_mode=False,
            )
            self.output_buffer = self.triangle_splatter.output_buffer

        elif discontinuity_mode == self.DiscontinuityMode.NVDR:
            self.antialiaser = tdr.Antialiaser(
                shade_buffer=self.silhouette_shader.output_buffer,
                depth_buffer=self.rasterizer.depth_buffer,
                triangle_id_buffer=self.rasterizer.triangle_id_buffer,
                clip_vertices=self.rasterizer.clip_vertices,
                triangle_vertex_ids=self.mesh.triangle_vertex_ids,
            )
            self.output_buffer = self.antialiaser.output_buffer

        else:
            raise Exception("discontinuity_mode must be one of [RTS, NVDR]")

        self.mse = tdr.ImageMeanSquaredError(
            self.output_buffer,
        )

    def forward(self):
        self.camera.forward()
        self.rasterizer.forward()
        self.silhouette_shader.forward()

        if self.discontinuity_mode == self.DiscontinuityMode.RTS:
            self.world_space_interpolator.forward()
            self.splat_position_interpolator.forward()
            self.triangle_splatter.forward()
        else:
            self.antialiaser.forward()

        self.mse.forward()

    def backward(self):
        self.mse.backward()

        if self.discontinuity_mode == self.DiscontinuityMode.RTS:
            self.triangle_splatter.backward()
            self.splat_position_interpolator.backward()
        else:
            self.antialiaser.backward()
            self.rasterizer.backward()

        self.camera.backward()

    def update(self):
        self.camera.update(self.lr_position, self.lr_rotation)


main()
