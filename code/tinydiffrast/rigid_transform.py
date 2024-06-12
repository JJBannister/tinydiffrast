import taichi as ti
import taichi.math as tm

from .triangle_mesh import TriangleMesh


@ti.data_oriented
class RigidTransform:
    """
    A learnable rigid body transform
    The representation of learnable rotations follows the approach proposed in Zhou and Barnes et al (2019)
    https://arxiv.org/pdf/1812.07035.pdf
    See equations 15 and 16 from supplemental section B for specifics w.r.t 3D rotation
    """

    def __init__(self):
        self.position_param = ti.Vector.field(
            n=3, dtype=float, shape=(), needs_grad=True
        )
        self.rotation_param = ti.Vector.field(
            n=3, dtype=float, shape=(2), needs_grad=True
        )
        self.matrix = ti.Matrix.field(4, 4, float, shape=(), needs_grad=True)

        # Initialize rotation as identity
        self.rotation_param[0].x = 1
        self.rotation_param[1].y = 1

    def forward(self):
        self.clear()
        self.compute_matrix()

    def backward(self):
        self.compute_matrix.grad()

    def clear(self):
        self.matrix.fill(0)
        self.matrix.grad.fill(0)
        self.position_param.grad.fill(0)
        self.rotation_param.grad.fill(0)

    @ti.kernel
    def update(self, lr_position: float, lr_rotation: float):
        self.position_param[None] -= lr_position * self.position_param.grad[None]
        for i in range(2):
            self.rotation_param[i] = tm.normalize(
                self.rotation_param[i] - lr_rotation * self.rotation_param.grad[i]
            )

        self.position_param.grad.fill(0)
        self.rotation_param.grad.fill(0)

    @ti.kernel
    def compute_matrix(self):
        a1 = self.rotation_param[0]
        a2 = self.rotation_param[1]
        position = self.position_param[None]

        b1 = tm.normalize(a1)
        b2 = tm.normalize(a2 - b1 * tm.dot(b1, a2))
        b3 = tm.cross(b1, b2)

        rotation_matrix = tm.mat4(
            [
                [b1[0], b2[0], b3[0], 0],
                [b1[1], b2[1], b3[1], 0],
                [b1[2], b2[2], b3[2], 0],
                [0, 0, 0, 1],
            ]
        )

        translation_matrix = tm.mat4(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [-position[0], -position[1], -position[2], 1],
            ]
        )

        self.matrix[None] = translation_matrix @ rotation_matrix


@ti.data_oriented
class RigidMeshTransform:

    def __init__(self, mesh: TriangleMesh, vertex_group: int = None):

        self.use_vertex_group = vertex_group is not None
        self.vertex_group = vertex_group

        self.mesh = mesh
        self.transformed_mesh = TriangleMesh(
            vertices=mesh.vertices.to_numpy(),
            triangle_vertex_ids=mesh.triangle_vertex_ids.to_numpy(),
            vertex_groups=mesh.vertex_groups.to_numpy(),
            triangle_groups=mesh.triangle_groups.to_numpy(),
        )

        self.rigid_transform = RigidTransform()

    def forward(self):
        self.clear()
        self.rigid_transform.forward()
        self.transform_vertices()

    def clear(self):
        self.rigid_transform.clear()
        self.transformed_mesh.vertices.grad.fill(0)

    def backward(self):
        self.transform_vertices.grad()
        self.rigid_transform.backward()

    def update(self):
        self.rigid_transform.update()

    @ti.kernel
    def transform_vertices(self):
        for i in range(self.mesh.n_vertices):
            v_group = self.mesh.vertex_groups[i]

            if ti.static(not self.use_vertex_group):
                vert = tm.vec4(self.mesh.vertices[i], 1)
                t_vert = vert @ self.rigid_transform.matrix[None]
                self.transformed_mesh.vertices[i] = t_vert.xyz

            elif v_group == self.vertex_group:
                vert = tm.vec4(self.mesh.vertices[i], 1)
                t_vert = vert @ self.rigid_transform.matrix[None]
                self.transformed_mesh.vertices[i] = t_vert.xyz

            else:
                self.transformed_mesh.vertices[i] = self.mesh.vertices[i]
