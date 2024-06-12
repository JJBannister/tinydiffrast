import unittest

import taichi as ti
import numpy as np

from tinydiffrast import TriangleMesh


class TestTriangleMesh(unittest.TestCase):

    def setUp(self) -> None:
        ti.init(arch=ti.cuda)

    def test_mesh(self):
        mesh = TriangleMesh.get_example_mesh("dragon")

        print()
        print("verts")
        print(mesh.vertices.shape)
        print(mesh.vertices.n)
        print(mesh.vertices.dtype)

        print()
        print("tris")
        print(mesh.triangle_vertex_ids.shape)
        print(mesh.triangle_vertex_ids.n)
        print(mesh.triangle_vertex_ids.dtype)

        print()
        print("vert groups")
        print(mesh.vertex_groups.shape)
        print(mesh.vertex_groups.dtype)

        print()
        print("tri groups")
        print(mesh.triangle_groups.shape)
        print(mesh.triangle_groups.dtype)


if __name__ == "__main__":
    unittest.main()
