import taichi as ti
import taichi.math as tm

from .triangle_mesh import TriangleMesh


@ti.data_oriented
class DirectionalLightArray:

    def __init__(
        self,
        n_lights: int = 1,
    ) -> None:

        assert n_lights > 0, "n_lights must be greater than 0"
        self.n_lights = n_lights

        self.light_directions = ti.Vector.field(
            n=3, dtype=float, shape=(n_lights), needs_grad=True
        )
        self.light_colors = ti.Vector.field(
            n=3, dtype=float, shape=(n_lights), needs_grad=True
        )

    def set_light_color(self, light_index: int, color: tm.vec3):
        assert (light_index < self.n_lights) and (
            light_index >= 0
        ), "light index is not valid"
        assert (color >= 0).all() and (color <= 1).all(), "Color must be in range [0,1]"
        self.light_colors[light_index] = color

    def set_light_direction(self, light_index: int, direction: tm.vec3):
        assert (light_index < self.n_lights) and (
            light_index >= 0
        ), "light index is not valid"
        self.light_directions[light_index] = direction / ti.sqrt((direction**2).sum())

    @ti.kernel
    def update(self, lr: float):
        for i in range(self.n_lights):
            self.light_colors[i] = self.light_colors[i] - lr * self.light_colors.grad[i]
            self.light_directions[i] = tm.normalize(
                self.light_directions[i] - lr * self.light_directions.grad[i]
            )
        self.light_colors.grad.fill(0)
        self.light_directions.grad.fill(0)
