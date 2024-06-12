import unittest

import taichi as ti
import numpy as np

import tinydiffrast as tdr


class TestTriangleInterpolators(unittest.TestCase):
    def setUp(self) -> None:
        ti.init(arch=ti.cuda, kernel_profiler=True)


    def test_triangle_interpolators(self):
        mesh = tdr.TriangleMesh.get_example_mesh("sphere_background")
        camera = tdr.PerspectiveCamera()
        camera.set_view_parameters(
            position=(0, 0, 3), focal_point=(0, 0, 0), up_vector=(0, 1, 0)
        )
        rasterizer = tdr.TriangleRasterizer((1024,1024), camera, mesh)

        world_space_interpolator = tdr.TriangleAttributeInterpolator(
            mesh.vertices,
            mesh.triangle_vertex_ids,
            rasterizer.triangle_id_buffer,
            rasterizer.barycentric_buffer
            )

        splat_position_interpolator = tdr.TriangleSplatPositionInterpolator(
            camera,
            world_space_interpolator.attribute_buffer,
            rasterizer.triangle_id_buffer
            )

        camera.forward()
        rasterizer.forward()
        world_space_interpolator.forward()
        splat_position_interpolator.forward()

        world_position_buffer = world_space_interpolator.attribute_buffer.to_numpy()
        tdr.plot_image(world_position_buffer[:,:,0,0], title="world position x")
        tdr.plot_image(world_position_buffer[:,:,0,1], title="world position y")
        tdr.plot_image(world_position_buffer[:,:,0,2], title="world position z")

        screen_space_buffer = splat_position_interpolator.screen_space_buffer.to_numpy()
        tdr.plot_image(screen_space_buffer[:,:,0,0], title="Splat Position X", colorbar=True)
        tdr.plot_image(screen_space_buffer[:,:,0,1], title="Splat Position Y", colorbar=True)

        for x in range(rasterizer.x_resolution):
            for y in range(rasterizer.y_resolution):
                for l in range(2):
                    if screen_space_buffer[x,y,l,0] != 0:
                        screen_space_buffer[x,y,l,0] -= x
                        screen_space_buffer[x,y,l,1] -= y

        print("Largest Screen Space Errors: ", np.max(screen_space_buffer), np.min(screen_space_buffer))
        tdr.plot_image(screen_space_buffer[:,:,0,0], title="X Error", colorbar=True)
        tdr.plot_image(screen_space_buffer[:,:,0,1], title="Y Error", colorbar=True)
        tdr.plot_image(screen_space_buffer[:,:,1,0], title="screen position x Error L1", colorbar=True)
        tdr.plot_image(screen_space_buffer[:,:,1,1], title="screen position y Error L1", colorbar=True)

        tdr.show_plots()


if __name__ == "__main__":
    unittest.main()