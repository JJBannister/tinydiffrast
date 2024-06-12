from typing import Tuple

import taichi as ti
import taichi.math as tm

from .triangle_mesh import TriangleMesh
from .cameras import Camera


# Define some custom vector dtypes
vec2 = ti.types.vector(2, ti.f32)
vec3 = ti.types.vector(3, ti.f32)
vec4 = ti.types.vector(4, ti.f32)

vec2_i32 = ti.types.vector(2, ti.i32)
vec3_i32 = ti.types.vector(3, ti.i32)
vec4_i32 = ti.types.vector(4, ti.i32)

vec2_i64 = ti.types.vector(2, ti.i64)
vec3_i64 = ti.types.vector(3, ti.i64)
vec4_i64 = ti.types.vector(4, ti.i64)


# Define some custom structs
@ti.dataclass
class AABB:
    x_min: ti.i32
    x_max: ti.i32
    y_min: ti.i32
    y_max: ti.i32

    @ti.func
    def is_point_inside(self, p: vec2_i32) -> bool:
        return (
            (p.x >= self.x_min)
            and (p.y >= self.y_min)
            and (p.x <= self.x_max)
            and (p.y <= self.y_max)
        )


@ti.dataclass
class SetupTriangle:
    triangle_id: ti.i32
    aabb: AABB

    origin_edge_values: vec3_i32
    pixel_edge_step_x: vec3_i32
    pixel_edge_step_y: vec3_i32

    depth_params: vec3
    max_z: ti.f32

    @ti.func
    def setup(
        self,
        triangle_id: ti.i32,
        v0_clip: vec4,
        v1_clip: vec4,
        v2_clip: vec4,
        x_resolution: ti.i32,
        y_resolution: ti.i32,
        subpixel_factor: ti.i32,
    ) -> bool:

        self.triangle_id = triangle_id
        enqueue = False

        # Agressive triangle culling because we do not implement clipping
        # Cull all triangles that are partially behind the camera
        if (v0_clip.w > 0) and (v1_clip.w > 0) and (v2_clip.w > 0):

            # Perspective Division
            v0_ndc = v0_clip.xyz / v0_clip.w
            v1_ndc = v1_clip.xyz / v1_clip.w
            v2_ndc = v2_clip.xyz / v2_clip.w

            # Map to screen coordinates
            v0_screen = ndc_to_screen(v0_ndc.xy, x_resolution, y_resolution)
            v1_screen = ndc_to_screen(v1_ndc.xy, x_resolution, y_resolution)
            v2_screen = ndc_to_screen(v2_ndc.xy, x_resolution, y_resolution)

            # Center the screen coordinates by applying an offset
            # This just gives a bit of extra room to avoid int overflow
            screen_origin = ti.cast(
                vec2([x_resolution // 2, y_resolution // 2]), ti.i32
            )
            v0_screen_offset = v0_screen - screen_origin
            v1_screen_offset = v1_screen - screen_origin
            v2_screen_offset = v2_screen - screen_origin

            # Multiply to adjust for subpixel precision
            v0_spp = tm.round(subpixel_factor * v0_screen_offset)
            v1_spp = tm.round(subpixel_factor * v1_screen_offset)
            v2_spp = tm.round(subpixel_factor * v2_screen_offset)

            # Compute 2x triangle area
            area_x2 = (v1_spp.x - v0_spp.x) * (v2_spp.y - v0_spp.y) - (
                v2_spp.x - v0_spp.x
            ) * (v1_spp.y - v0_spp.y)

            # Cull backfacing and degenerate triangles
            if area_x2 > 0:

                # Compute AABB in screen coordinates
                self.aabb = AABB(
                    x_min=tm.floor(tm.min(v0_screen.x, v1_screen.x, v2_screen.x)),
                    x_max=tm.ceil(tm.max(v0_screen.x, v1_screen.x, v2_screen.x)),
                    y_min=tm.floor(tm.min(v0_screen.y, v1_screen.y, v2_screen.y)),
                    y_max=tm.ceil(tm.max(v0_screen.y, v1_screen.y, v2_screen.y)),
                )

                # Convert vertices to integer dtypes
                # Use 64 bits during setup to avoid int overflow
                v0_i = ti.cast(v0_spp, ti.i64)
                v1_i = ti.cast(v1_spp, ti.i64)
                v2_i = ti.cast(v2_spp, ti.i64)

                # Compute edge function coefficients
                subpixel_edge_step_x = vec3_i64(
                    [
                        v1_i.y - v2_i.y,
                        v2_i.y - v0_i.y,
                        v0_i.y - v1_i.y,
                    ]
                )
                subpixel_edge_step_y = vec3_i64(
                    [
                        v2_i.x - v1_i.x,
                        v0_i.x - v2_i.x,
                        v1_i.x - v0_i.x,
                    ]
                )

                # Evaluate the edge functions at the origin (0,0) of the sub-pixel coordinate system
                # This is the step that requires 64 bits to avoid int overflow
                spp_origin_edge_values = vec3_i64(
                    [
                        v1_i.x * v2_i.y - v1_i.y * v2_i.x,
                        v2_i.x * v0_i.y - v2_i.y * v0_i.x,
                        v0_i.x * v1_i.y - v0_i.y * v1_i.x,
                    ]
                )

                # Adjustments for top-left fill rule
                for i in range(3):

                    # In a CCW triangle, a top edge is an edge that is exactly horizontal and goes towards the left
                    if (subpixel_edge_step_x[i] == 0) and (subpixel_edge_step_y[i] < 0):
                        continue

                    # In a CCW triangle, a left edge is an edge that goes down
                    elif subpixel_edge_step_x[i] > 0:
                        continue

                    # Otherwise, apply a bias to non top-left edges
                    else:
                        spp_origin_edge_values[i] -= 1

                # We can now toss away the sub-pixel bits and switch to 32 bit integers.
                # This is possible because we will always take pixel sized steps in our rasterizer.
                # Therefore, the sub-pixel bits will be exactly the same at every pixel and there is no point keeping these useless bits around anymore.
                spp_origin_edge_values = ti.cast(
                    spp_origin_edge_values // subpixel_factor, ti.i32
                )

                # Evaluate the edge values at the actual origin in screen space
                self.origin_edge_values = ti.cast(
                    spp_origin_edge_values
                    - subpixel_edge_step_x * screen_origin.x
                    - subpixel_edge_step_y * screen_origin.y,
                    ti.i32,
                )

                # Again, discarding sub-pixel bits by using sub-pixel step sizes as pixel step sizes
                # Implicitly, we are multiplying and dividing by the sub-pixel factor
                self.pixel_edge_step_x = ti.cast(subpixel_edge_step_x, ti.i32)
                self.pixel_edge_step_y = ti.cast(subpixel_edge_step_y, ti.i32)

                # Finally, compute depth params
                self.depth_params = tm.vec3(
                    [
                        v0_ndc.z,
                        (v1_ndc.z - v0_ndc.z) / area_x2,
                        (v2_ndc.z - v0_ndc.z) / area_x2,
                    ]
                )
                self.max_z = tm.max(v0_ndc.z, v1_ndc.z, v2_ndc.z)

                # If we made it this far, the triangle should be added to raster queues
                enqueue = True
        return enqueue

    @ti.func
    def evaluate_edge_functions(self, pixel: vec2_i32) -> vec3_i32:
        return (
            self.origin_edge_values
            + pixel.x * self.pixel_edge_step_x
            + pixel.y * self.pixel_edge_step_y
        )

    @ti.func
    def interpolate_depth(self, edge_values: vec3_i32) -> float:
        return (
            self.depth_params.x
            + self.depth_params.y * float(edge_values[1])
            + self.depth_params.z * float(edge_values[2])
        )


# Define some helper functions
@ti.func
def ndc_to_screen(v: vec2, x_resolution: int, y_resolution: int) -> vec2:
    return vec2(
        [
            0.5 * (x_resolution * v.x + x_resolution - 1),
            0.5 * (y_resolution * v.y + y_resolution - 1),
        ]
    )


@ti.func
def screen_to_ndc(p: vec2_i32, x_resolution: int, y_resolution: int) -> vec2:
    x_resolution = ti.f32(x_resolution)
    y_resolution = ti.f32(y_resolution)
    p = ti.cast(p, ti.f32)

    return tm.vec2(
        [
            (2 * p.x - x_resolution + 1) / x_resolution,
            (2 * p.y - y_resolution + 1) / y_resolution,
        ]
    )


@ti.func
def do_aabbs_overlap(aabb1: AABB, aabb2: AABB) -> bool:
    return (
        (aabb1.x_min <= aabb2.x_max)
        and (aabb1.y_min <= aabb2.y_max)
        and (aabb2.x_min <= aabb1.x_max)
        and (aabb2.y_min <= aabb1.y_max)
    )


@ti.func
def compute_barycentrics_and_depth(
    pixel: vec2_i32,
    x_resolution: int,
    y_resolution: int,
    v0: vec4,
    v1: vec4,
    v2: vec4,
) -> Tuple[vec2, float]:

    # Get pixel position in clip/ndc space
    p = screen_to_ndc(pixel, x_resolution, y_resolution)

    # Compute edge functions
    d0 = v0.xy - p * v0.w
    d1 = v1.xy - p * v1.w
    d2 = v2.xy - p * v2.w

    a0 = d1.x * d2.y - d1.y * d2.x
    a1 = d2.x * d0.y - d2.y * d0.x
    a2 = d0.x * d1.y - d0.y * d1.x

    # Perspective correction and normalization
    sum_inv = 1.0 / (a0 + a1 + a2)
    barys = vec3([a0 * sum_inv, a1 * sum_inv, a2 * sum_inv])

    # Perspective correct depth
    z = v0.z * barys[0] + v1.z * barys[1] + v2.z * barys[2]
    w = v0.w * barys[0] + v1.w * barys[1] + v2.w * barys[2]
    zw = z / w

    return [barys.xy, zw]


@ti.data_oriented
class TriangleRasterizer:
    def __init__(
        self,
        resolution: Tuple[int, int],
        camera: Camera,
        mesh: TriangleMesh,
        subpixel_bits: int = 8,
    ):

        self.camera = camera
        self.mesh = mesh
        self.n_triangles = mesh.n_triangles
        self.n_vertices = mesh.n_vertices
        self.subpixel_factor = 2**subpixel_bits

        # Only support xy resolutions that match tile/bin size
        self.n_pixels_per_tile = 8
        self.n_tiles_per_bin = 16
        self.n_pixels_per_bin = self.n_pixels_per_tile * self.n_tiles_per_bin

        x_resolution, y_resolution = resolution
        assert (x_resolution % self.n_pixels_per_bin == 0) and (
            x_resolution > 0
        ), "x_resolution must be a positive integer multiple of bin size {}".format(
            self.n_pixels_per_bin
        )
        assert (y_resolution % self.n_pixels_per_bin == 0) and (
            y_resolution > 0
        ), "y_resolution must be a positive integer multiple of bin size {}".format(
            self.n_pixels_per_bin
        )

        self.x_resolution = x_resolution
        self.y_resolution = y_resolution

        self.n_x_tiles = int(x_resolution / self.n_pixels_per_tile)
        self.n_y_tiles = int(y_resolution / self.n_pixels_per_tile)
        self.n_x_bins = int(self.n_x_tiles / self.n_tiles_per_bin)
        self.n_y_bins = int(self.n_y_tiles / self.n_tiles_per_bin)

        # Create data containers
        self.clip_vertices = ti.Vector.field(
            n=4, dtype=float, shape=self.n_vertices, needs_grad=True
        )
        self.setup_triangles = SetupTriangle.field(shape=self.n_triangles)

        self.tile_raster_queue = ti.field(int)
        ti.root.dense(ti.ij, (self.n_x_tiles, self.n_y_tiles)).dynamic(
            ti.k, self.n_triangles, chunk_size=32
        ).place(self.tile_raster_queue)

        self.triangle_id_buffer = ti.field(int)
        self.depth_buffer = ti.field(float)
        self.barycentric_buffer = ti.Vector.field(2, float, needs_grad=True)

        # Layout pixel fields with tile/pixel/layer structure
        pixel_root = (
            ti.root.dense(ti.ij, (self.n_x_tiles, self.n_y_tiles))
            .dense(ti.ij, (8, 8))
            .dense(ti.k, 2)
        )
        pixel_root.place(
            self.triangle_id_buffer, self.depth_buffer, self.barycentric_buffer
        )
        pixel_root.place(self.barycentric_buffer.grad)

    def forward(self):
        self.clear()
        self.vertex_setup()
        self.triangle_setup()
        self.rasterize()
        self.compute_barycentrics()

    def backward(self):
        self.compute_barycentrics.grad()
        self.vertex_setup.grad()

    @ti.kernel
    def clear(self):
        self.triangle_id_buffer.fill(0)
        self.depth_buffer.fill(0)
        self.barycentric_buffer.fill(0)

        self.clip_vertices.grad.fill(0)
        self.barycentric_buffer.grad.fill(0)
        for x, y in ti.ndrange(self.n_x_tiles, self.n_y_tiles):
            self.tile_raster_queue[x, y].deactivate()

            # This causes the queue to alocate a block of memory
            # It improves performance for scenes with very large triangles where allocations would otherwise happen in sequence
            self.tile_raster_queue[x, y].append(0)

    @ti.kernel
    def vertex_setup(self):
        for i in range(self.n_vertices):
            self.clip_vertices[i] = (
                vec4(self.mesh.vertices[i], 1)
                @ self.camera.view_matrix[None]
                @ self.camera.projection_matrix[None]
            )

    @ti.kernel
    def triangle_setup(self):
        for i in range(self.n_triangles):
            verts_ids = self.mesh.triangle_vertex_ids[i] - 1
            v0 = self.clip_vertices[verts_ids[0]]
            v1 = self.clip_vertices[verts_ids[1]]
            v2 = self.clip_vertices[verts_ids[2]]

            enqueue = self.setup_triangles[i].setup(
                i + 1,
                v0,
                v1,
                v2,
                self.x_resolution,
                self.y_resolution,
                self.subpixel_factor,
            )

            if enqueue:
                self.enqueue_triangle(i, self.setup_triangles[i])

    @ti.kernel
    def rasterize(self):
        for i, j in ti.ndrange(self.n_x_tiles, self.n_y_tiles):
            self.rasterize_tile(i, j)

    @ti.kernel
    def compute_barycentrics(self):
        for i, j, k in self.triangle_id_buffer:
            tri_id = self.triangle_id_buffer[i, j, k]
            if tri_id != 0:

                # Grab clip vertex coords for the relevant triangle
                verts_ids = self.mesh.triangle_vertex_ids[tri_id - 1] - 1
                v0 = self.clip_vertices[verts_ids[0]]
                v1 = self.clip_vertices[verts_ids[1]]
                v2 = self.clip_vertices[verts_ids[2]]

                # Compute perspective correct barycentrics and depth
                barys, depth = compute_barycentrics_and_depth(
                    tm.vec2([i, j]), self.x_resolution, self.y_resolution, v0, v1, v2
                )

                # Populate buffers
                self.barycentric_buffer[i, j, k] = barys
                self.depth_buffer[i, j, k] = depth

    @ti.func
    def enqueue_triangle(self, triangle_id: int, triangle: SetupTriangle) -> None:
        # Check bins for occupancy
        for i, j in ti.ndrange(self.n_x_bins, self.n_y_bins):
            bin_aabb = AABB(
                x_min=i * self.n_pixels_per_bin,
                x_max=(i + 1) * self.n_pixels_per_bin,
                y_min=j * self.n_pixels_per_bin,
                y_max=(j + 1) * self.n_pixels_per_bin,
            )
            triangle_is_in_bin = do_aabbs_overlap(triangle.aabb, bin_aabb)

            # Check tiles in bin for occupancy
            if triangle_is_in_bin:
                for k, l in ti.ndrange(self.n_tiles_per_bin, self.n_tiles_per_bin):
                    tile_aabb = AABB(
                        x_min=bin_aabb.x_min + k * self.n_pixels_per_tile,
                        x_max=bin_aabb.x_min + (k + 1) * self.n_pixels_per_tile,
                        y_min=bin_aabb.y_min + l * self.n_pixels_per_tile,
                        y_max=bin_aabb.y_min + (l + 1) * self.n_pixels_per_tile,
                    )
                    triangle_is_in_tile = do_aabbs_overlap(triangle.aabb, tile_aabb)

                    # Enqueue the triangle
                    if triangle_is_in_tile:
                        tile_x = i * self.n_tiles_per_bin + k
                        tile_y = j * self.n_tiles_per_bin + l
                        self.tile_raster_queue[tile_x, tile_y].append(triangle_id)

    @ti.func
    def rasterize_tile(self, tile_x: int, tile_y: int) -> None:
        tile_aabb = AABB(
            x_min=tile_x * self.n_pixels_per_tile,
            x_max=(tile_x + 1) * self.n_pixels_per_tile,
            y_min=tile_y * self.n_pixels_per_tile,
            y_max=(tile_y + 1) * self.n_pixels_per_tile,
        )

        for k in range(self.tile_raster_queue[tile_x, tile_y].length()):
            # Ignore the first element that was placed just to trigger memory allocation
            if k == 0:
                continue

            triangle = self.setup_triangles[self.tile_raster_queue[tile_x, tile_y, k]]

            # Iterate over pixels that are in both tile and triangle aabb
            intersect_aabb = AABB(
                x_min=tm.max(tile_aabb.x_min, triangle.aabb.x_min),
                x_max=tm.min(tile_aabb.x_max, triangle.aabb.x_max + 1),
                y_min=tm.max(tile_aabb.y_min, triangle.aabb.y_min),
                y_max=tm.min(tile_aabb.y_max, triangle.aabb.y_max + 1),
            )

            # Setup for incremental edge evaluation
            origin_edge_values = triangle.evaluate_edge_functions(
                vec2_i32([intersect_aabb.x_min, intersect_aabb.y_min])
            )
            column_edge_values = origin_edge_values
            edge_values = origin_edge_values

            for x in range(intersect_aabb.x_min, intersect_aabb.x_max):
                for y in range(intersect_aabb.y_min, intersect_aabb.y_max):
                    if (edge_values >= 0).all():

                        # Early depth kill
                        l1_depth = self.depth_buffer[x, y, 1]
                        if triangle.max_z > l1_depth:

                            # Z buffering
                            depth = triangle.interpolate_depth(edge_values)
                            if (depth >= 0) and (depth <= 1):

                                l0_empty = self.triangle_id_buffer[x, y, 0] == 0
                                l0_depth = self.depth_buffer[x, y, 0]

                                if l0_empty:
                                    self.triangle_id_buffer[x, y, 0] = (
                                        triangle.triangle_id
                                    )
                                    self.depth_buffer[x, y, 0] = depth

                                elif depth > l0_depth:
                                    self.triangle_id_buffer[x, y, 1] = (
                                        self.triangle_id_buffer[x, y, 0]
                                    )
                                    self.depth_buffer[x, y, 1] = self.depth_buffer[
                                        x, y, 0
                                    ]

                                    self.triangle_id_buffer[x, y, 0] = (
                                        triangle.triangle_id
                                    )
                                    self.depth_buffer[x, y, 0] = depth

                                elif depth > l1_depth:
                                    self.triangle_id_buffer[x, y, 1] = (
                                        triangle.triangle_id
                                    )
                                    self.depth_buffer[x, y, 1] = depth

                    # Increment edge values by y step
                    edge_values += triangle.pixel_edge_step_y

                # Increment edge values by x step
                column_edge_values += triangle.pixel_edge_step_x
                edge_values = column_edge_values
