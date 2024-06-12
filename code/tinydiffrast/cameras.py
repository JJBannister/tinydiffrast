from typing import Tuple
from abc import abstractmethod, ABC

import taichi as ti
import taichi.math as tm
import numpy as np

from .rigid_transform import RigidTransform


@ti.data_oriented
class Camera(ABC):

    def __init__(self):
        self.rigid_transform = RigidTransform()
        self.view_matrix = self.rigid_transform.matrix
        self.projection_matrix = ti.Matrix.field(4, 4, float, shape=())

        self.set_view_parameters(
            position=(0, 0, 4), focal_point=(0, 0, 0), up_vector=(0, 1, 0)
        )
        self.set_plane_parameters(near_plane=1, far_plane=10)

    def forward(self):
        self.clear()
        self.rigid_transform.forward()
        self.compute_projection_matrix()

    def backward(self):
        self.rigid_transform.backward()

    def clear(self):
        self.projection_matrix.fill(0)

    def update(self, lr_position: float, lr_rotation: float):
        self.rigid_transform.update(lr_position, lr_rotation)

    def set_view_parameters(
        self,
        position: Tuple[float, float, float],
        focal_point: Tuple[float, float, float],
        up_vector: Tuple[float, float, float],
    ):
        position = np.array(position, dtype=np.float32)
        focal_point = np.array(focal_point, dtype=np.float32)
        up_vector = np.array(up_vector, dtype=np.float32)

        z_vec = self.normalize(position - focal_point)
        x_vec = self.normalize(np.cross(up_vector, z_vec))
        y_vec = self.normalize(np.cross(z_vec, x_vec))

        self.rigid_transform.position_param[None] = position
        self.rigid_transform.rotation_param[0] = x_vec
        self.rigid_transform.rotation_param[1] = y_vec

    def set_plane_parameters(self, near_plane: float, far_plane: float):
        assert near_plane > 0, "near_plane must be positive"
        assert far_plane > 0, "far_plane must be positive"
        assert far_plane > near_plane, "far_plane must be greater than near_plane"

        self.near_plane = near_plane
        self.far_plane = far_plane

    @staticmethod
    def normalize(np_vector: np.array) -> np.array:
        return np_vector / np.linalg.norm(np_vector)

    @abstractmethod
    @ti.kernel
    def compute_projection_matrix(self):
        pass


class PerspectiveCamera(Camera):
    def __init__(self):
        super().__init__()
        self.fov = ti.Vector.field(n=2, dtype=float, shape=())
        self.set_field_of_view(60, 60)

    def set_field_of_view(self, horizontal_fov: float, vertical_fov: float):
        """FOV parameters are in units of degrees."""
        assert (
            horizontal_fov > 0 and horizontal_fov < 180
        ), "horizontal_fov must be in range (0,180) degrees"
        assert (
            vertical_fov > 0 and vertical_fov < 180
        ), "vertical_fov must be in range (0,180) degrees"

        self.horizontal_fov = horizontal_fov
        self.vertical_fov = vertical_fov

    @ti.kernel
    def compute_projection_matrix(self):
        m00 = 1.0 / tm.tan(tm.pi * self.horizontal_fov / 360.0)
        m11 = 1.0 / tm.tan(tm.pi * self.vertical_fov / 360.0)
        m22 = self.near_plane / (self.far_plane - self.near_plane)
        m32 = (self.far_plane * self.near_plane) / (self.far_plane - self.near_plane)

        self.projection_matrix[None] = tm.mat4(
            [[m00, 0, 0, 0], [0, m11, 0, 0], [0, 0, m22, -1], [0, 0, m32, 0]]
        )


class OrthographicCamera(Camera):
    def __init__(self):
        super().__init__()
        self.projection_size = ti.Vector.field(n=2, dtype=float, shape=())
        self.set_projection_size(5, 5)

    def set_projection_size(self, projection_width: float, projection_height: float):
        assert projection_width > 0, "projection width must be > 0"
        assert projection_height > 0, "projection height must be > 0"

        self.projection_width = projection_width
        self.projection_height = projection_height

    @ti.kernel
    def compute_projection_matrix(self):
        width = self.projection_width
        height = self.projection_height
        near_plane = self.near_plane
        far_plane = self.far_plane

        m00 = 2.0 / width
        m11 = 2.0 / height
        m22 = 1.0 / (far_plane - near_plane)
        m32 = far_plane / (far_plane - near_plane)

        self.projection_matrix[None] = tm.mat4(
            [[m00, 0, 0, 0], [0, m11, 0, 0], [0, 0, m22, 0], [0, 0, m32, 1]]
        )
