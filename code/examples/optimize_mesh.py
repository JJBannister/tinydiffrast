from enum import Enum

import taichi as ti
import taichi.math as tm
import numpy as np

ti.init(arch=ti.cuda)

import tinydiffrast as tdr


def main():
    # Create ground truth image
    true_pipe = Pipeline(
        resolution=(1024, 1024),
        discontinuity_mode=Pipeline.DiscontinuityMode.RTS,
        mesh="bunny",
    )
    true_pipe.forward()
    true_image = true_pipe.output_buffer.to_numpy()

    # Create trainable pipeline
    pipe = Pipeline(
        resolution=(1024, 1024),
        mesh="icosphere",
        discontinuity_mode=Pipeline.DiscontinuityMode.RTS,
        regularization_weight=0.05,
    )
    pipe.mse.set_reference_image(true_image)
    pipe.forward()
    initial_image = pipe.output_buffer.to_numpy()

    # Pre-Train Visualize
    tdr.plot_image(true_image, title="True Image")
    tdr.plot_image(initial_image, title="Initial Image")
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
            mse_loss = pipe.mse.loss[None]
            regularizer_loss = pipe.regularizer.loss[None]
            print(
                "Train step: {}, mse_loss: {}, regularizer_loss: {}".format(
                    i, mse_loss, regularizer_loss
                )
            )

        if i % 500 == 0:
            pipe.regularizer.set_reference_edge_lengths()

    # Post-Train Visualize
    final_image = pipe.output_buffer.to_numpy()
    tdr.plot_image(true_image, title="True Image")
    tdr.plot_image(initial_image, title="Initial Image")
    tdr.plot_image(final_image, title="Final Image")
    tdr.show_plots()


class Pipeline:

    class DiscontinuityMode(Enum):
        RTS = 1
        NVDR = 2

    def __init__(
        self,
        resolution=(1024, 1024),
        mesh="dragon",
        discontinuity_mode=DiscontinuityMode.RTS,
        regularization_weight=1.0,
    ):

        self.discontinuity_mode = discontinuity_mode
        self.resolution = resolution
        self.lr = 1e0

        self.mesh = tdr.TriangleMesh.get_example_mesh(mesh)
        self.camera = tdr.PerspectiveCamera()
        self.rasterizer = tdr.TriangleRasterizer(resolution, self.camera, self.mesh)

        self.shader = tdr.SilhouetteShader(
            self.rasterizer.triangle_id_buffer,
            triangle_groups=self.mesh.triangle_groups,
        )

        if discontinuity_mode == self.DiscontinuityMode.RTS:
            self.world_position_interpolator = tdr.TriangleAttributeInterpolator(
                vertex_attributes=self.mesh.vertices,
                triangle_vertex_ids=self.mesh.triangle_vertex_ids,
                triangle_id_buffer=self.rasterizer.triangle_id_buffer,
                barycentric_buffer=self.rasterizer.barycentric_buffer,
            )
            self.splat_position_interpolator = tdr.TriangleSplatPositionInterpolator(
                camera=self.camera,
                world_space_buffer=self.world_position_interpolator.attribute_buffer,
                triangle_id_buffer=self.rasterizer.triangle_id_buffer,
            )
            self.triangle_splatter = tdr.TriangleSplatter(
                shade_buffer=self.shader.output_buffer,
                screen_space_buffer=self.splat_position_interpolator.screen_space_buffer,
                depth_buffer=self.rasterizer.depth_buffer,
                triangle_id_buffer=self.rasterizer.triangle_id_buffer,
                single_layer_mode=False,
            )

            self.output_buffer = self.triangle_splatter.output_buffer

        elif discontinuity_mode == self.DiscontinuityMode.NVDR:
            self.antialiaser = tdr.Antialiaser(
                shade_buffer=self.shader.output_buffer,
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
        self.regularizer = tdr.EdgeLengthRegularizer(
            mesh=self.mesh, weight=regularization_weight
        )

    def forward(self):
        self.camera.forward()
        self.rasterizer.forward()
        self.shader.forward()

        if self.discontinuity_mode == self.DiscontinuityMode.RTS:
            self.world_position_interpolator.forward()
            self.splat_position_interpolator.forward()
            self.triangle_splatter.forward()
        else:
            self.antialiaser.forward()

        self.mse.forward()
        self.regularizer.forward()

    def backward(self):
        self.mse.backward()
        self.regularizer.backward()

        if self.discontinuity_mode == self.DiscontinuityMode.RTS:
            self.triangle_splatter.backward()
            self.splat_position_interpolator.backward()
            self.world_position_interpolator.backward()
        else:
            self.antialiaser.backward()
            self.rasterizer.backward()

    def update(self):
        self.mesh.update(self.lr)


main()
