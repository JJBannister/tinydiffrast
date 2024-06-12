import unittest

import numpy as np
import taichi as ti

import tinydiffrast as tdr


class TestOcclusionEstimator(unittest.TestCase):
    def setUp(self) -> None:
        ti.init(arch=ti.cuda, kernel_profiler=True)


    def test_occlusion_estimator(self):
        mesh = tdr.TriangleMesh.get_example_mesh("sphere_background")
        camera = tdr.PerspectiveCamera()

        rasterizer = tdr.TriangleRasterizer((128,128), camera, mesh, subpixel_bits=4)
        occlusion_estimator = tdr.OcclusionEstimator(
            depth_buffer=rasterizer.depth_buffer,
            triangle_id_buffer=rasterizer.triangle_id_buffer,
            background_occlusion=True)

        camera.forward()
        rasterizer.forward()
        occlusion_estimator.forward()

        triangle_id_buffer = rasterizer.triangle_id_buffer.to_numpy()[:,:,0]
        is_occluded_buffer = occlusion_estimator.is_occluded_buffer.to_numpy()
        is_occluder_buffer = occlusion_estimator.is_occluder_buffer.to_numpy()

        tdr.plot_image(triangle_id_buffer, title="Triangle id buffer")
        tdr.plot_image(is_occluder_buffer, title="Is Occluder")
        tdr.plot_image(is_occluded_buffer, title="Is Occluded")
        tdr.show_plots()


if __name__ == "__main__":
    unittest.main()