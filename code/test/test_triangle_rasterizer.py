import unittest

import taichi as ti
import numpy as np

from tinydiffrast import (
    TriangleMesh,
    TriangleRasterizer,
    OrthographicCamera,
    PerspectiveCamera,
    plot_image,
    show_plots,
)


class TestTriangleRasterizer(unittest.TestCase):
    def setUp(self) -> None:
        ti.init(arch=ti.cpu, kernel_profiler=True)

    def test_profile_rasterizer(self):
        mesh = TriangleMesh.get_example_mesh("dragon")
        camera = PerspectiveCamera()
        rasterizer = TriangleRasterizer((1024, 1024), camera, mesh)

        camera.forward()
        for _ in range(1000):
            rasterizer.forward()
            rasterizer.backward()

        ti.profiler.print_kernel_profiler_info()

        depth_buffer = rasterizer.depth_buffer.to_numpy()
        plot_image(depth_buffer[:, :, 0], title="Z Layer 0", colorbar=True)
        show_plots()

    def test_rasterize(self):
        mesh = TriangleMesh.get_example_mesh("sphere")
        camera = OrthographicCamera()
        rasterizer = TriangleRasterizer((1024, 1024), camera, mesh)

        camera.forward()
        rasterizer.forward()

        depth_buffer = rasterizer.depth_buffer.to_numpy()
        tri_buffer = rasterizer.triangle_id_buffer.to_numpy()
        bary_buffer = rasterizer.barycentric_buffer.to_numpy()

        plot_image(depth_buffer[:, :, 0], title="Z Layer 0", colorbar=True)
        plot_image(depth_buffer[:, :, 1], title="Z Layer 1", colorbar=True)

        plot_image(tri_buffer[:, :, 0], title="Triangle Id Layer 0")
        plot_image(tri_buffer[:, :, 1], title="Triangle Id Layer 1")

        plot_image(
            bary_buffer[:, :, 0, 0], title="Barycentric Layer 0 U", colorbar=True
        )
        plot_image(
            bary_buffer[:, :, 0, 1], title="Barycentric Layer 0 V", colorbar=True
        )
        plot_image(
            bary_buffer[:, :, 1, 0], title="Barycentric Layer 1 U", colorbar=True
        )
        plot_image(
            bary_buffer[:, :, 1, 1], title="Barycentric Layer 1 V", colorbar=True
        )

        show_plots()

    def test_rasterize_grad(self):
        mesh = TriangleMesh.get_example_mesh("sphere")
        camera = OrthographicCamera()
        rasterizer = TriangleRasterizer((1024, 1024), camera, mesh)

        camera.forward()
        rasterizer.forward()

        rasterizer.barycentric_buffer.grad.fill(1)
        rasterizer.backward()

        vert_grads = mesh.vertices.grad.to_numpy()
        print(np.min(vert_grads), np.max(vert_grads))


if __name__ == "__main__":
    unittest.main()
