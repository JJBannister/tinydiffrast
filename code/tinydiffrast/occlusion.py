import taichi as ti


@ti.data_oriented
class OcclusionEstimator:

    def __init__(
        self,
        depth_buffer: ti.Vector.field,
        triangle_id_buffer: ti.field,
        background_occlusion: bool = False,
    ) -> None:

        self.depth_buffer = depth_buffer
        self.triangle_id_buffer = triangle_id_buffer
        self.background_occlusion = background_occlusion

        self.x_resolution = depth_buffer.shape[0]
        self.y_resolution = depth_buffer.shape[1]

        self.is_occluded_buffer = ti.field(
            dtype=int, shape=(self.x_resolution, self.y_resolution)
        )
        self.is_occluder_buffer = ti.field(
            dtype=int, shape=(self.x_resolution, self.y_resolution)
        )

    def forward(self):
        self.clear()
        self.compute_occlusions()

    def clear(self):
        self.is_occluded_buffer.fill(0)
        self.is_occluder_buffer.fill(0)

    @ti.func
    def is_valid_pixel(self, x: int, y: int) -> bool:
        return (
            (x >= 0)
            and (x < self.x_resolution)
            and (y >= 0)
            and (y < self.y_resolution)
        )

    @ti.kernel
    def compute_occlusions(self):
        for x, y in ti.ndrange(self.x_resolution, self.y_resolution):

            # Check if the pixel is empty
            top_is_active = self.triangle_id_buffer[x, y, 0] != 0
            bottom_is_active = self.triangle_id_buffer[x, y, 1] != 0

            if top_is_active:

                # for each pixel in a 3x3 neighborhood
                for l in range(3):
                    for m in range(3):
                        neighbor_x = x + l - 1
                        neighbor_y = y + m - 1

                        # Check if the neighbor is outside the bounds of the image
                        if not self.is_valid_pixel(neighbor_x, neighbor_y):
                            continue

                        # Check if the neighbor is empty
                        neighbor_is_active = (
                            self.triangle_id_buffer[neighbor_x, neighbor_y, 0] != 0
                        )
                        if (not neighbor_is_active) and self.background_occlusion:
                            self.is_occluder_buffer[x, y] = 1
                            self.is_occluded_buffer[neighbor_x, neighbor_y] = 1
                            continue

                        # Check if the bottom layer of neighbor pixel is active or empty
                        neighbor_bottom_is_active = (
                            self.triangle_id_buffer[neighbor_x, neighbor_y, 1] != 0
                        )

                        # Find the layers with the best match in terms of depth
                        # If the second layers are empty they cannot be matched with
                        d00 = ti.abs(
                            self.depth_buffer[x, y, 0]
                            - self.depth_buffer[neighbor_x, neighbor_y, 0]
                        )
                        d01 = (
                            ti.abs(
                                self.depth_buffer[x, y, 0]
                                - self.depth_buffer[neighbor_x, neighbor_y, 1]
                            )
                            if neighbor_bottom_is_active
                            else float("inf")
                        )
                        d10 = (
                            ti.abs(
                                self.depth_buffer[x, y, 1]
                                - self.depth_buffer[neighbor_x, neighbor_y, 0]
                            )
                            if bottom_is_active
                            else float("inf")
                        )

                        # The top layers are the best match
                        # No occlusion
                        if (d00 < d01) and (d00 < d10):
                            continue

                        # The splat occludes the neighbor
                        if d10 < d01:
                            self.is_occluder_buffer[x, y] = 1

                        # The neighbor occludes the splat
                        if d01 < d10:
                            self.is_occluded_buffer[x, y] = 1
