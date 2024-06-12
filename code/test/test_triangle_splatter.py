import unittest

import numpy as np
import taichi as ti

import tinydiffrast as tdr


class TestTriangleSplatter(unittest.TestCase):
    def setUp(self) -> None:
        ti.init(arch=ti.cuda, kernel_profiler=True)


    def test_triangle_splatter(self):
        mesh = tdr.TriangleMesh.get_example_mesh("sphere_background")
        camera = tdr.PerspectiveCamera()
        camera.set_view_parameters(
            position=(0, 0, 3), focal_point=(0, 0, 0), up_vector=(0, 1, 0)
        )
        rasterizer = tdr.TriangleRasterizer((128,128), camera, mesh, subpixel_bits=4)

        world_space_interpolator = tdr.TriangleAttributeInterpolator(
            mesh.vertices,
            mesh.triangle_vertex_ids,
            rasterizer.triangle_id_buffer,
            rasterizer.barycentric_buffer)

        splat_position_interpolator = tdr.TriangleSplatPositionInterpolator(
            camera,
            world_space_interpolator.attribute_buffer,
            rasterizer.triangle_id_buffer,
            perfect_coords=True)

        triangle_shader = tdr.SilhouetteShader(
            rasterizer.triangle_id_buffer,
            triangle_groups=mesh.triangle_groups)

        triangle_splatter = tdr.TriangleSplatter(
            triangle_shader.output_buffer,
            splat_position_interpolator.screen_space_buffer,
            rasterizer.depth_buffer,
            rasterizer.triangle_id_buffer,
            single_layer_mode=False)

        camera.forward()
        rasterizer.forward()
        world_space_interpolator.forward()
        splat_position_interpolator.forward()
        triangle_shader.forward()
        triangle_splatter.forward()

        self.viz_splats(triangle_splatter)


    def viz_splats(self, triangle_splatter):

        depth_buffer = triangle_splatter.depth_buffer.to_numpy()
        color_buffer = triangle_splatter.shade_buffer.to_numpy()
        splat_weight_buffer = triangle_splatter.splat_weight_buffer.to_numpy()
        output_buffer = triangle_splatter.output_buffer.to_numpy()
        is_occluded_buffer = triangle_splatter.occlusion_estimator.is_occluded_buffer.to_numpy()
        is_occluder_buffer = triangle_splatter.occlusion_estimator.is_occluder_buffer.to_numpy()

        tdr.plot_image(is_occluded_buffer, title="Is Occluded")
        tdr.plot_image(is_occluder_buffer, title="Is Occluder")

        tdr.plot_image(splat_weight_buffer[:,:,2], title="s- weights", cmap='jet', colorbar=True, vmin=0, vmax=1.1)
        tdr.plot_image(splat_weight_buffer[:,:,1], title="s0 weights", cmap='jet', colorbar=True, vmin=0, vmax=1.1)
        tdr.plot_image(splat_weight_buffer[:,:,0], title="s+ weights", cmap='jet', colorbar=True, vmin=0, vmax=1.1)

        tdr.plot_image(output_buffer[:,:], title="Splat Output")

        tdr.plot_image(depth_buffer[:,:,1], title="Depth L1", cmap='jet', colorbar=True)
        tdr.plot_image(depth_buffer[:,:,0], title="Depth L0", cmap='jet', colorbar=True)

        tdr.show_plots()


if __name__ == "__main__":
    unittest.main()