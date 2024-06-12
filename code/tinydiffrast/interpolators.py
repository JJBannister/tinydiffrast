import taichi as ti
import taichi.math as tm

from .cameras import Camera
from .rasterizer import ndc_to_screen


@ti.data_oriented
class TriangleAttributeInterpolator:

    def __init__(
        self,
        vertex_attributes: ti.Vector.field,
        triangle_vertex_ids: ti.Vector.field,
        triangle_id_buffer: ti.Vector.field,
        barycentric_buffer: ti.Vector.field,
        normalize: bool = False,
    ) -> None:

        self.vertex_attributes = vertex_attributes
        self.triangle_vertex_ids = triangle_vertex_ids
        self.triangle_id_buffer = triangle_id_buffer
        self.barycentric_buffer = barycentric_buffer
        self.normalize = normalize

        self.attribute_buffer = ti.Vector.field(
            n=vertex_attributes.n,
            dtype=float,
            shape=triangle_id_buffer.shape,
            needs_grad=True,
        )

    def forward(self):
        self.clear()
        self.interpolate()

    def backward(self):
        self.interpolate.grad()

    def clear(self):
        self.attribute_buffer.fill(0)
        self.attribute_buffer.grad.fill(0)

    @ti.kernel
    def interpolate(self):
        for x, y, l in self.attribute_buffer:

            triangle_id = self.triangle_id_buffer[x, y, l]
            if triangle_id != 0:

                # Get attributes
                vert_ids = self.triangle_vertex_ids[triangle_id - 1] - 1
                a0 = self.vertex_attributes[vert_ids[0]]
                a1 = self.vertex_attributes[vert_ids[1]]
                a2 = self.vertex_attributes[vert_ids[2]]

                # Get barycentrics
                barys = self.barycentric_buffer[x, y, l]
                b0 = barys[0]
                b1 = barys[1]
                b2 = 1.0 - (b0 + b1)

                # Interpolate
                a = b0 * a0 + b1 * a1 + b2 * a2
                if self.normalize: a = tm.normalize(a)
                self.attribute_buffer[x, y, l] = a


@ti.data_oriented
class TriangleSplatPositionInterpolator:
    """
    Interpolates screen space coordinates needed for computing splat positions
    """

    def __init__(
        self,
        camera: Camera,
        world_space_buffer: ti.Vector.field,
        triangle_id_buffer: ti.Vector.field,
        perfect_coords: bool = False,
    ) -> None:

        self.camera = camera
        self.world_space_buffer = world_space_buffer
        self.triangle_id_buffer = triangle_id_buffer

        self.perfect_coords = perfect_coords

        self.screen_space_buffer = ti.Vector.field(
            n=2, dtype=float, shape=self.world_space_buffer.shape, needs_grad=True
        )

        self.x_resolution = triangle_id_buffer.shape[0]
        self.y_resolution = triangle_id_buffer.shape[1]

    def forward(self):
        self.clear()
        self.interpolate()

    def backward(self):
        self.interpolate.grad()

    def clear(self):
        self.screen_space_buffer.fill(0)
        self.screen_space_buffer.grad.fill(0)

    @ti.kernel
    def interpolate(self):
        for x, y, l in self.screen_space_buffer:
            triangle_id = self.triangle_id_buffer[x, y, l]

            if triangle_id != 0:

                if self.perfect_coords:
                    self.screen_space_buffer[x, y, l] = x, y

                else:
                    world_coordinates = tm.vec4(self.world_space_buffer[x, y, l], 1)
                    view_coordinates = world_coordinates @ self.camera.view_matrix[None]
                    clip_coordinates = (
                        view_coordinates @ self.camera.projection_matrix[None]
                    )
                    ndc_coordinates = clip_coordinates.xy / clip_coordinates.w
                    self.screen_space_buffer[x, y, l] = ndc_to_screen(
                        ndc_coordinates, self.x_resolution, self.y_resolution
                    )
