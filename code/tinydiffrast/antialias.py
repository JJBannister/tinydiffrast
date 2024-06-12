from typing import Tuple

import taichi as ti
import taichi.math as tm

from .occlusion import OcclusionEstimator
from .triangle_mesh import TriangleMesh 
from .rasterizer import ndc_to_screen, vec2_i32


@ti.data_oriented
class Antialiaser:

    def __init__(
        self,
        shade_buffer: ti.Vector.field,
        depth_buffer: ti.field,
        triangle_id_buffer: ti.field,
        clip_vertices: ti.Vector.field,
        triangle_vertex_ids: ti.Vector.field,
    ):
        """
        Implementation of the nvdiffrast antialiasing operation:
        https://nvlabs.github.io/nvdiffrast/
        """

        self.shade_buffer = shade_buffer
        self.triangle_id_buffer = triangle_id_buffer

        self.clip_vertices = clip_vertices
        self.triangle_vertex_ids = triangle_vertex_ids

        self.x_resolution = shade_buffer.shape[0]
        self.y_resolution = shade_buffer.shape[1]

        self.occlusion_estimator = OcclusionEstimator(
            depth_buffer=depth_buffer,
            triangle_id_buffer=triangle_id_buffer,
            background_occlusion=True,
        )

        # The blend factor field stores how much each occluder pixel is blended with neighbor pixels
        # The 4 values per pixel correspond to neighbors in the order of [up, down, left, right]
        self.blend_factor_buffer = ti.field(
            dtype=float,
            shape=(self.x_resolution, self.y_resolution, 4),
            needs_grad=True,
        )

        self.output_buffer = ti.Vector.field(
            n=3,
            dtype=float,
            shape=(self.x_resolution, self.y_resolution),
            needs_grad=True,
        )

    def forward(self):
        self.clear()
        self.occlusion_estimator.forward()
        self.compute_blend_factors()
        self.blend()

    def backward(self):
        self.blend.grad()
        self.compute_blend_factors.grad()

    def clear(self):
        self.output_buffer.fill(0)
        self.blend_factor_buffer.fill(0)
        self.output_buffer.grad.fill(0)
        self.blend_factor_buffer.grad.fill(0)

    @ti.kernel
    def compute_blend_factors(self):
        for x, y in ti.ndrange(self.x_resolution, self.y_resolution):

            pix_is_occluder = self.occlusion_estimator.is_occluder_buffer[x, y]
            if pix_is_occluder:

                # Get vertices in screen space
                triangle_id = self.triangle_id_buffer[x, y, 0]
                vert_ids = self.triangle_vertex_ids[triangle_id - 1] - 1

                v0_clip = self.clip_vertices[vert_ids[0]]
                v1_clip = self.clip_vertices[vert_ids[1]]
                v2_clip = self.clip_vertices[vert_ids[2]]

                v0_screen = ndc_to_screen(
                    v0_clip.xy / v0_clip.w, self.x_resolution, self.y_resolution
                )
                v1_screen = ndc_to_screen(
                    v1_clip.xy / v1_clip.w, self.x_resolution, self.y_resolution
                )
                v2_screen = ndc_to_screen(
                    v2_clip.xy / v2_clip.w, self.x_resolution, self.y_resolution
                )

                # Compute and set the blend factors for each neighbor
                for i in ti.static(range(4)):
                    self.compute_blend_factor(
                        v0_screen, v1_screen, v2_screen, vec2_i32([x, y]), i
                    )

    @ti.kernel
    def blend(self):
        for x, y in self.output_buffer:
            pix_is_occluder = self.occlusion_estimator.is_occluder_buffer[x, y]
            pix_is_occluded = self.occlusion_estimator.is_occluded_buffer[x, y]

            # If pix is neither occluded nor occluder, pass it through untouched
            if (not pix_is_occluded) and (not pix_is_occluder):
                self.output_buffer[x, y] = self.shade_buffer[x, y, 0]

            else:
                # Grab the blend factors
                up_blend = self.blend_factor_buffer[x, y, 0]
                down_blend = self.blend_factor_buffer[x, y, 1]
                left_blend = self.blend_factor_buffer[x, y, 2]
                right_blend = self.blend_factor_buffer[x, y, 3]

                # Normalize in rare case blend factors sum to > 1
                blend_sum = up_blend + down_blend + left_blend + right_blend
                blend_normalization = 1.0 if blend_sum < 1 else blend_sum
                pixel_blend = 0.0 if blend_sum > 1 else 1.0 - blend_sum

                # Blend
                pixel_color = self.shade_buffer[x, y, 0]
                up_color = self.shade_buffer[x, y + 1, 0]
                down_color = self.shade_buffer[x, y - 1, 0]
                left_color = self.shade_buffer[x - 1, y, 0]
                right_color = self.shade_buffer[x + 1, y, 0]

                output_color = (
                    pixel_color * pixel_blend
                    + up_color * up_blend / blend_normalization
                    + down_color * down_blend / blend_normalization
                    + left_color * left_blend / blend_normalization
                    + right_color * right_blend / blend_normalization
                )

                self.output_buffer[x, y] = output_color

    @ti.func
    def compute_blend_factor(
        self, v0: tm.vec2, v1: tm.vec2, v2: tm.vec2, pixel: vec2_i32, neighbor_index: int
    ) -> None:

        pixel_index = 0
        neighbor = vec2_i32(0)
        if neighbor_index == 0:
            neighbor = vec2_i32([pixel[0], pixel[1] + 1])
            pixel_index = 1

        elif neighbor_index == 1:
            neighbor = vec2_i32([pixel[0], pixel[1] - 1])
            pixel_index = 0

        elif neighbor_index == 2:
            neighbor = vec2_i32([pixel[0] - 1, pixel[1]])
            pixel_index = 3

        elif neighbor_index == 3:
            neighbor = vec2_i32([pixel[0] + 1, pixel[1]])
            pixel_index = 2

        is_valid = False
        edge_crossing = 0.0
        # Is the neighbor pixel occluded?
        if self.occlusion_estimator.is_occluded_buffer[neighbor[0], neighbor[1]]:

            # Is the neighbor pixel valid?
            if self.occlusion_estimator.is_valid_pixel(neighbor[0], neighbor[1]):

                pixel_vec = tm.vec2(pixel)
                neighbor_vec = tm.vec2(neighbor)

                # Find the relevant triangle edge and compute the edge crossing
                # Try e0
                if not is_valid:
                    e0_is_oriented = self.are_lines_perpindicular(
                        pixel_vec, neighbor_vec, v0, v1
                    )
                    if e0_is_oriented:
                        e0_is_valid, e0_crossing = self.compute_edge_crossing(
                            pixel_vec, neighbor_vec, v0, v1
                        )
                        if e0_is_valid:
                            is_valid = True
                            edge_crossing = e0_crossing

                # Try e1
                if not is_valid:
                    e1_is_oriented = self.are_lines_perpindicular(
                        pixel_vec, neighbor_vec, v1, v2
                    )
                    if e1_is_oriented:
                        e1_is_valid, e1_crossing = self.compute_edge_crossing(
                            pixel_vec, neighbor_vec, v1, v2
                        )
                        if e1_is_valid:
                            is_valid = True
                            edge_crossing = e1_crossing

                # Try e2
                if not is_valid:
                    e2_is_oriented = self.are_lines_perpindicular(
                        pixel_vec, neighbor_vec, v2, v0
                    )
                    if e2_is_oriented:
                        e2_is_valid, e2_crossing = self.compute_edge_crossing(
                            pixel_vec, neighbor_vec, v2, v0
                        )
                        if e2_is_valid:
                            is_valid = True
                            edge_crossing = e2_crossing

        if is_valid:
            # Depending on the location of the edge crossing
            # Blend the neighbor into the pixel or blend the pixel into the neighbor
            blend_factor = ti.abs(0.5 - edge_crossing)
            if edge_crossing <= 0.5:
                self.blend_factor_buffer[pixel[0], pixel[1], neighbor_index] = (
                    blend_factor
                )
            else:
                self.blend_factor_buffer[neighbor[0], neighbor[1], pixel_index] = (
                    blend_factor
                )

    @staticmethod
    @ti.func
    def are_lines_perpindicular(
        p1: tm.vec2, p2: tm.vec2, p3: tm.vec2, p4: tm.vec2
    ) -> bool:
        """Returns true iff the angle subtended by line segments p1-p2 and p3-p4 is > 45 degrees"""
        vec_a = ti.abs(tm.normalize(p2 - p1))
        vec_b = ti.abs(tm.normalize(p4 - p3))
        return tm.dot(vec_a, vec_b) < tm.cos(tm.pi / 4.0)

    @staticmethod
    @ti.func
    def compute_edge_crossing(
        p1: tm.vec2, p2: tm.vec2, p3: tm.vec2, p4: tm.vec2
    ) -> Tuple[bool, float]:
        """
        Computes the intersection of a triangle edge line segment
        with the line segment connecting the centers of a pixel and it's neighbor

        Following notation/formulae from https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
        p1 -> p2 is the line segment from the center of the pixel to the center of the neighbor pixel
        p3 -> p4 is the triangle edge line segment

        The normalized edge crossing (t) is the first order bezier parameter of p1 -> p2
        (u) is the first order bezier parameter of p3 -> p4
        Both (t) and (u) must be in range [0,1] for an intersection to be valid
        """
        t = ((p1.x - p3.x) * (p3.y - p4.y) - (p1.y - p3.y) * (p3.x - p4.x)) / (
            (p1.x - p2.x) * (p3.y - p4.y) - (p1.y - p2.y) * (p3.x - p4.x)
        )
        u = ((p1.x - p3.x) * (p1.y - p2.y) - (p1.y - p3.y) * (p1.x - p2.x)) / (
            (p1.x - p2.x) * (p3.y - p4.y) - (p1.y - p2.y) * (p3.x - p4.x)
        )
        is_valid = (t >= 0) and (t <= 1) and (u >= 0) and (u <= 1)
        return [is_valid, t]
