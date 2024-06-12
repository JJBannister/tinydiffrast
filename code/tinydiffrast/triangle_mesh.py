from importlib import resources

import taichi as ti
import taichi.math as tm
import numpy as np

from . import mesh_data


@ti.data_oriented
class TriangleMesh:
    def __init__(
        self,
        vertices: np.array,
        triangle_vertex_ids: np.array,
        vertex_groups: np.array = None,
        triangle_groups: np.array = None,
    ):

        self.n_vertices = vertices.shape[0]
        self.n_triangles = triangle_vertex_ids.shape[0]

        self.vertices = ti.Vector.field(
            n=3, dtype=float, shape=self.n_vertices, needs_grad=True
        )
        self.triangle_vertex_ids = ti.Vector.field(
            n=3, dtype=int, shape=self.n_triangles
        )

        self.vertices.from_numpy(vertices)
        self.triangle_vertex_ids.from_numpy(triangle_vertex_ids)

        if vertex_groups is not None:
            self.vertex_groups = ti.field(dtype=int, shape=self.n_vertices)
            self.vertex_groups.from_numpy(vertex_groups)

        if triangle_groups is not None:
            self.triangle_groups = ti.field(dtype=int, shape=self.n_triangles)
            self.triangle_groups.from_numpy(triangle_groups)

    @ti.kernel
    def update(self, lr: float):
        for v in range(self.n_vertices):
            self.vertices[v] -= lr * self.vertices.grad[v]
        self.vertices.grad.fill(0.)            

    @staticmethod
    def get_example_mesh(mesh_shape="cube"):
        """Loads an example mesh"""
        file_path = resources.files(mesh_data) / "{}.obj".format(mesh_shape)
        return TriangleMesh.load_triangle_mesh_from_obj(file_path)

    @staticmethod
    def load_triangle_mesh_from_obj(obj_file_path: str):
        vertices = []
        vertex_groups = []
        triangle_vertex_ids = []
        triangle_groups = []

        active_group = -1
        with open(obj_file_path) as file:
            for line in file:
                line = line.rstrip().split(" ")

                # Parse object identifier
                if line[0] == "o":
                    active_group += 1

                # Parse vertices
                elif line[0] == "v":
                    vertices.append([line[1], line[2], line[3]])
                    vertex_groups.append(max(0, active_group))

                # Parse faces
                elif line[0] == "f":
                    if len(line) != 4:
                        raise Exception("This mesh contains non-traingular faces.")

                    v1 = line[1].split("/")
                    v2 = line[2].split("/")
                    v3 = line[3].split("/")

                    triangle_vertex_ids.append([v1[0], v2[0], v3[0]])
                    triangle_groups.append(max(0, active_group))

        # Create a TriangleMesh object
        vertices = np.array(vertices, dtype=np.float32)
        triangle_vertex_ids = np.array(triangle_vertex_ids, dtype=np.int32)
        vertex_groups = np.array(vertex_groups, dtype=np.int32)
        triangle_groups = np.array(triangle_groups, dtype=np.int32)

        return TriangleMesh(
            vertices=vertices,
            triangle_vertex_ids=triangle_vertex_ids,
            vertex_groups=vertex_groups,
            triangle_groups=triangle_groups,
        )
