from math import exp

import taichi as ti
import taichi.math as tm
import numpy as np

from .interpolators import TriangleSplatPositionInterpolator
from .occlusion import OcclusionEstimator


@ti.data_oriented
class TriangleSplatter:

    def __init__(
        self,
        shade_buffer: ti.Vector.field,
        screen_space_buffer: ti.Vector.field,
        depth_buffer: ti.field,
        triangle_id_buffer: ti.field,
        kernel_variance: float = 0.25,
        adjustment_factor: float = 0.05,
        single_layer_mode: bool = False,
    ):
        """
        Implementation of the splatting operation described in Cole et al. 2021
        https://arxiv.org/abs/2108.04886
        """

        self.single_layer_mode = single_layer_mode

        self.shade_buffer = shade_buffer
        self.screen_space_buffer = screen_space_buffer
        self.depth_buffer = depth_buffer
        self.triangle_id_buffer = triangle_id_buffer

        self.kernel_variance = kernel_variance
        self.x_resolution = shade_buffer.shape[0]
        self.y_resolution = shade_buffer.shape[1]

        self.occlusion_estimator = OcclusionEstimator(
            depth_buffer=depth_buffer, triangle_id_buffer=triangle_id_buffer
        )

        # Create fields
        self.splat_buffer = ti.Vector.field(
            n=3,
            dtype=float,
            shape=(self.x_resolution, self.y_resolution, 3),
            needs_grad=True,
        )

        self.splat_weight_buffer = ti.field(
            dtype=float,
            shape=(self.x_resolution, self.y_resolution, 3),
            needs_grad=True,
        )

        self.output_buffer = ti.Vector.field(
            n=3,
            dtype=float,
            shape=(self.x_resolution, self.y_resolution),
            needs_grad=True,
        )

        # Pre-compute normalization constant for gaussian splats
        kernel_sum = 0.0
        for u in [-1, 0, 1]:
            for v in [-1, 0, 1]:
                d_squared = u**2 + v**2
                kernel_sum += exp(-d_squared / (2.0 * kernel_variance))
        self.normalization_constant = (1.0 + adjustment_factor) / kernel_sum

    def forward(self):
        self.clear()
        self.occlusion_estimator.forward()
        self.splat()
        self.normalize_and_composite()

    def backward(self):
        self.normalize_and_composite.grad()
        self.splat.grad()

    def clear(self):
        self.splat_weight_buffer.fill(0)
        self.splat_buffer.fill(0)
        self.output_buffer.fill(0)
        self.splat_buffer.grad.fill(0)
        self.splat_weight_buffer.grad.fill(0)
        self.output_buffer.grad.fill(0)

    @ti.kernel
    def splat(self):
        for x, y, l in self.shade_buffer:
            if self.triangle_id_buffer[x, y, l] != 0:

                # Grab the xy splat location and color
                splat_location = self.screen_space_buffer[x, y, l]
                splat_color = self.shade_buffer[x, y, l]

                # Splat over the 3x3 neighborhood
                for i in ti.static(range(3)):
                    for j in ti.static(range(3)):
                        neighbor_x = x + i - 1
                        neighbor_y = y + j - 1

                        # Check if the neighbor is outside the bounds of the image
                        if self.occlusion_estimator.is_valid_pixel(
                            neighbor_x, neighbor_y
                        ):

                            pix_is_occluder = (
                                self.occlusion_estimator.is_occluder_buffer[x, y]
                            )
                            pix_is_occluded = (
                                self.occlusion_estimator.is_occluded_buffer[x, y]
                            )
                            neighbor_is_occluder = (
                                self.occlusion_estimator.is_occluder_buffer[
                                    neighbor_x, neighbor_y
                                ]
                            )
                            neighbor_is_occluded = (
                                self.occlusion_estimator.is_occluded_buffer[
                                    neighbor_x, neighbor_y
                                ]
                            )

                            # Default splatting layer for no occlusion
                            splat_layer = l + 1

                            # Both pixels are occluders or both pixels are occluded is treated as no occlusion
                            if (pix_is_occluded and neighbor_is_occluded) or (
                                pix_is_occluder and neighbor_is_occluder
                            ):
                                pass

                            # Pixel occludes the neighbor
                            elif pix_is_occluder and neighbor_is_occluded:
                                splat_layer = l

                            # Pixel is occluded by the neighbor
                            elif pix_is_occluded and neighbor_is_occluder:
                                splat_layer = 2

                            # Splat
                            d_squared = (splat_location.x - float(neighbor_x)) ** 2 + (
                                splat_location.y - float(neighbor_y)
                            ) ** 2
                            w = self.splat_weight(d_squared)

                            # If single layer mode, just ignore the second layer and all occlusion information
                            if self.single_layer_mode:
                                if l == 0:
                                    self.splat_weight_buffer[
                                        neighbor_x, neighbor_y, 1
                                    ] += w
                                    self.splat_buffer[neighbor_x, neighbor_y, 1] += (
                                        w * splat_color
                                    )

                            else:
                                self.splat_weight_buffer[
                                    neighbor_x, neighbor_y, splat_layer
                                ] += w
                                self.splat_buffer[
                                    neighbor_x, neighbor_y, splat_layer
                                ] += (w * splat_color)

    @ti.kernel
    def normalize_and_composite(self):
        for x, y in ti.ndrange(self.x_resolution, self.y_resolution):

            # Extract weights
            l0_weight = self.splat_weight_buffer[x, y, 0]
            l1_weight = self.splat_weight_buffer[x, y, 1]
            l2_weight = self.splat_weight_buffer[x, y, 2]

            # Extract colors
            l0_color = self.splat_buffer[x, y, 0]
            l1_color = self.splat_buffer[x, y, 1]
            l2_color = self.splat_buffer[x, y, 2]

            # Normalize if required
            if l0_weight > 1:
                l0_color = l0_color / l0_weight
                l0_weight = 1.0

            if l1_weight > 1:
                l1_color = l1_color / l1_weight
                l1_weight = 1.0

            if l2_weight > 1:
                l2_color = l2_color / l2_weight
                l2_weight = 1.0

            # Composite l0 over l1 over l2
            self.output_buffer[x, y] = l0_color + (1.0 - l0_weight) * (
                l1_color + (1.0 - l1_weight) * l2_color
            )

    @ti.func
    def splat_weight(self, d_squared: float) -> float:
        """
        Evaluates the gaussian splat kernel at distance d from the splat center
        Includes the adjustment factor and normalization by the kernel sum
        """
        return (
            tm.exp(-d_squared / (2.0 * self.kernel_variance))
            * self.normalization_constant
        )
