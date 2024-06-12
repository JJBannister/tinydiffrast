from typing import Tuple

import numpy as np
import taichi as ti
import taichi.math as tm

from .triangle_mesh import TriangleMesh

@ti.data_oriented
class ImageMeanSquaredError:
    def __init__(
        self,
        image_buffer: ti.Vector.field,
    ):

        self.image_buffer = image_buffer

        image_shape = image_buffer.shape
        self.n_pixels = float(image_shape[0] * image_shape[1])

        self.reference_buffer = ti.Vector.field(
            n=image_buffer.n, dtype=float, shape=image_shape
        )
        self.loss = ti.field(float, shape=(), needs_grad=True)

    def forward(self):
        self.clear()
        self.compute_loss()

    def clear(self):
        self.loss.fill(0)
        self.loss.grad.fill(0)

    def backward(self):
        self.loss.grad.fill(1.0)
        self.compute_loss.grad()

    def set_reference_image(self, reference_image: np.array):
        self.reference_buffer.from_numpy(reference_image)

    @ti.kernel
    def compute_loss(self):
        for i, j in self.image_buffer:
            true_pixel = self.image_buffer[i, j]
            reference_pixel = self.reference_buffer[i, j]
            self.loss[None] += (
                (true_pixel - reference_pixel) ** 2
            ).sum() / self.n_pixels


@ti.data_oriented
class EdgeLengthRegularizer:

    def __init__(self, mesh: TriangleMesh, weight: float = 1.0):
        self.mesh = mesh
        self.weight = ti.field(dtype=float, shape=())
        self.weight[None] = weight

        # Allocate fields
        self.loss = ti.field(dtype=float, shape=(), needs_grad=True)

        # Build data structures for vertex-vertex connections
        self.build_connections()

        # Initialize the edge lengths used as reference
        self.set_reference_edge_lengths()

    def forward(self):
        self.clear()
        self.compute_loss()

    def clear(self):
        self.loss.fill(0)

    def backward(self):
        self.loss.grad.fill(1.0)
        self.compute_loss.grad()

    @ti.kernel
    def compute_loss(self):
        for e in self.edges:
            v0 = self.mesh.vertices[self.edges[e][0]]
            v1 = self.mesh.vertices[self.edges[e][1]]
            length = tm.distance(v0, v1)
            reference_length = self.edge_lengths[e]
            self.loss[None] += self.weight[None]*(length - reference_length) ** 2

    @ti.kernel
    def set_reference_edge_lengths(self):
        for e in self.edges:
            v0 = self.mesh.vertices[self.edges[e][0]]
            v1 = self.mesh.vertices[self.edges[e][1]]
            self.edge_lengths[e] = tm.distance(v0, v1)

    def build_connections(self):
        # Using raw python here because I don't want to manually implement a hash set in Taichi just for this function
        edges = set()
        for t in range(self.mesh.n_triangles):
            verts_ids = self.mesh.triangle_vertex_ids[t] - 1
            edges.add(frozenset([verts_ids[0], verts_ids[1]]))
            edges.add(frozenset([verts_ids[1], verts_ids[2]]))
            edges.add(frozenset([verts_ids[2], verts_ids[0]]))

        # Store the edges as a field
        n_edges = len(edges)

        self.edges = ti.Vector.field(n=2, dtype=int, shape=(n_edges))
        self.edges.from_numpy(np.array([list(e) for e in edges]))

        # Create another field to store the edge lengths
        self.edge_lengths = ti.field(dtype=float, shape=(n_edges))
