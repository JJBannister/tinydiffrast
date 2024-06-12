import taichi as ti
import taichi.math as tm

from .triangle_mesh import TriangleMesh


@ti.data_oriented
class NormalsEstimator:

    def __init__(
        self,
        mesh: TriangleMesh,
    ) -> None:

        self.mesh = mesh

        self.accumulation_field = ti.Vector.field(
            n=3, dtype=float, shape=(mesh.n_vertices), needs_grad=True
        )

        self.normals = ti.Vector.field(
            n=3, dtype=float, shape=(mesh.n_vertices), needs_grad=True
        )

    def forward(self):
        self.clear()
        self.accumulate_normals()
        self.normalize()

    def clear(self):
        self.accumulation_field.fill(0)
        self.normals.fill(0)
        self.accumulation_field.grad.fill(0)
        self.normals.grad.fill(0)

    def backward(self):
        self.normalize.grad()
        self.accumulate_normals.grad()

    @ti.kernel
    def accumulate_normals(self):
        for i in range(self.mesh.n_triangles):
            verts_ids = self.mesh.triangle_vertex_ids[i] - 1
            v0 = self.mesh.vertices[verts_ids[0]]
            v1 = self.mesh.vertices[verts_ids[1]]
            v2 = self.mesh.vertices[verts_ids[2]]

            face_normal = tm.normalize(tm.cross(v1 - v0, v2 - v0))
            angle_0 = tm.acos(tm.dot(tm.normalize(v1 - v0), tm.normalize(v2 - v0)))
            angle_1 = tm.acos(tm.dot(tm.normalize(v0 - v1), tm.normalize(v2 - v1)))
            angle_2 = tm.acos(tm.dot(tm.normalize(v0 - v2), tm.normalize(v1 - v2)))

            self.accumulation_field[verts_ids[0]] += angle_0 * face_normal
            self.accumulation_field[verts_ids[1]] += angle_1 * face_normal
            self.accumulation_field[verts_ids[2]] += angle_2 * face_normal

    @ti.kernel
    def normalize(self):
        for i in range(self.mesh.n_vertices):
            self.normals[i] = tm.normalize(self.accumulation_field[i])
