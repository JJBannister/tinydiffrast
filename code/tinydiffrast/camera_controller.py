from typing import Tuple
import time

import taichi as ti
import taichi.math as tm
import numpy as np

from .cameras import Camera


class CameraController:

    def __init__(
        self,
        camera: Camera,
        position_sensitivity: float = 1.7,
        rotation_sensitivity: float = 0.4,
    ):
        self.camera = camera
        self.rotation_sensitivity = rotation_sensitivity
        self.position_sensitivity = position_sensitivity
        self.last_time = None

    def get_view_basis(self) -> Tuple[np.array, np.array, np.array]:
        a1 = self.camera.rigid_transform.rotation_param[0].to_numpy()
        a2 = self.camera.rigid_transform.rotation_param[1].to_numpy()

        b1 = self.camera.normalize(a1)
        b2 = self.camera.normalize(a2 - b1 * np.dot(b1, a2))
        b3 = np.cross(b1, b2)
        return (b1, b2, b3)

    def track_inputs(self, window: ti.ui.Window) -> None:
        # Use elapsed time to calibrate movement speed
        if self.last_time is None:
            self.last_time = time.perf_counter_ns()
        time_elapsed = (time.perf_counter_ns() - self.last_time) * 1e-9
        self.last_time = time.perf_counter_ns()

        position_speed = self.position_sensitivity * time_elapsed
        rotation_speed = self.rotation_sensitivity * time_elapsed

        # camera basis vectors
        x, y, z = self.get_view_basis()

        # Update camera position
        position_delta = np.zeros(3)

        # Zoom in/out
        if window.is_pressed("e"):
            position_delta -= position_speed * z
        if window.is_pressed("q"):
            position_delta += position_speed * z

        # Pan up/down/left/right
        if window.is_pressed("w"):
            position_delta += position_speed * y
        if window.is_pressed("s"):
            position_delta -= position_speed * y
        if window.is_pressed("a"):
            position_delta -= position_speed * x
        if window.is_pressed("d"):
            position_delta += position_speed * x

        self.camera.rigid_transform.position_param[None] += position_delta

        # Update camera rotation
        x_delta = np.zeros(3)
        y_delta = np.zeros(3)

        # Rotate camera (pitch)
        if window.is_pressed(ti.ui.UP):
            y_delta += rotation_speed * z
        if window.is_pressed(ti.ui.DOWN):
            y_delta -= rotation_speed * z

        # Rotate camera (yaw)
        if window.is_pressed(ti.ui.RIGHT):
            x_delta += rotation_speed * z
        if window.is_pressed(ti.ui.LEFT):
            x_delta -= rotation_speed * z

        # Rotate camera (roll)
        if window.is_pressed("z"):
            x_delta += rotation_speed * y
            y_delta -= rotation_speed * x
        if window.is_pressed("x"):
            x_delta -= rotation_speed * y
            y_delta += rotation_speed * x

        self.camera.rigid_transform.rotation_param[0] = tm.vec3(
            self.camera.normalize(x_delta + x)
        )
        self.camera.rigid_transform.rotation_param[1] = tm.vec3(
            self.camera.normalize(y_delta + y)
        )
