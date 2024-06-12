import unittest

import taichi as ti
import taichi.math as tm

ti.init(arch=ti.cuda)

from tinydiffrast import OrthographicCamera, PerspectiveCamera


class TestCameras(unittest.TestCase):

    def setUp(self) -> None:
        self.position = [0, 0, -5]
        self.focal_point = [0, 0, 0]
        self.up_vector = [0, 1, 0]
        self.near_plane = 3
        self.far_plane = 10

    def test_orthographic_camera(self):
        cam = OrthographicCamera()
        cam.set_view_parameters(
            position=self.position,
            focal_point=self.focal_point,
            up_vector=self.up_vector,
        )
        cam.set_plane_parameters(near_plane=self.near_plane, far_plane=self.far_plane)
        cam.set_projection_size(projection_width=5, projection_height=5)

        cam.forward()
        m_view = cam.view_matrix[None]
        m_proj = cam.projection_matrix[None]
        cam.backward()

        print("Orthographic Camera")
        print("view matrix")
        print(m_view)
        print("projection matrix")
        print(m_proj)
        print()
        print()

    def test_Perspective_camera(self):
        cam = PerspectiveCamera()
        cam.set_view_parameters(
            position=self.position,
            focal_point=self.focal_point,
            up_vector=self.up_vector,
        )
        cam.set_plane_parameters(near_plane=self.near_plane, far_plane=self.far_plane)
        cam.set_field_of_view(horizontal_fov=50, vertical_fov=50)

        cam.forward()
        m_view = cam.view_matrix[None]
        m_proj = cam.projection_matrix[None]
        cam.backward()

        print()
        print("Perspective Camera ")
        print("view matrix ")
        print(m_view)
        print("projection matrix ")
        print(m_proj)
        print()


if __name__ == "__main__":
    unittest.main()
