import taichi as ti
import taichi.math as tm

ti.init(arch=ti.cuda)

import tinydiffrast as tdr


def main():
    pipeline = Pipeline(resolution=(1024, 1024), mesh_type="bunny")

    window = ti.ui.Window("Interactive Renderer", res=pipeline.output_buffer.shape)
    canvas = window.get_canvas()
    controller = tdr.CameraController(pipeline.camera)

    print()
    print("*** Running Interactive Renderer ***")
    print()
    print("'wasd' keys control camera xy translation")
    print("'eq' keys control camera z translation")
    print("Arrow keys control camera pitch and yaw")
    print("'zx' keys control camera roll")
    print()
    print("'kl' keys control the visible layer")
    print()

    while window.running:
        pipeline.forward()
        canvas.set_image(pipeline.output_buffer)
        window.show()

        controller.track_inputs(window)
        if window.get_event(ti.ui.PRESS):
            match window.event.key:
                case "k":
                    pipeline.active_layer[None] = 0
                    print("Showing Layer 0")
                case "l":
                    pipeline.active_layer[None] = 1
                    print("Showing Layer 1")


@ti.data_oriented
class Pipeline:

    def __init__(self, resolution=(1024, 1024), mesh_type="dragon"):

        self.mesh = tdr.TriangleMesh.get_example_mesh(mesh_type)
        self.camera = tdr.PerspectiveCamera()

        self.rasterizer = tdr.TriangleRasterizer(
            resolution, self.camera, self.mesh, subpixel_bits=4
        )
        self.world_position_interpolator = tdr.TriangleAttributeInterpolator(
            vertex_attributes=self.mesh.vertices,
            triangle_vertex_ids=self.mesh.triangle_vertex_ids,
            triangle_id_buffer=self.rasterizer.triangle_id_buffer,
            barycentric_buffer=self.rasterizer.barycentric_buffer,
        )

        self.normals_estimator = tdr.NormalsEstimator(mesh=self.mesh)
        self.normals_interpolator = tdr.TriangleAttributeInterpolator(
            vertex_attributes=self.normals_estimator.normals,
            triangle_vertex_ids=self.mesh.triangle_vertex_ids,
            triangle_id_buffer=self.rasterizer.triangle_id_buffer,
            barycentric_buffer=self.rasterizer.barycentric_buffer,
        )

        self.vertex_albedos = ti.Vector.field(
            n=3, dtype=float, shape=(self.mesh.n_vertices), needs_grad=True
        )
        self.vertex_albedos.fill(1)
        self.albedo_interpolator = tdr.TriangleAttributeInterpolator(
            vertex_attributes=self.vertex_albedos,
            triangle_vertex_ids=self.mesh.triangle_vertex_ids,
            triangle_id_buffer=self.rasterizer.triangle_id_buffer,
            barycentric_buffer=self.rasterizer.barycentric_buffer,
        )

        self.light_array = tdr.DirectionalLightArray(n_lights=3)
        self.light_array.set_light_color(0, tm.vec3([1, 1, 1]))
        self.light_array.set_light_direction(0, tm.vec3([0, 0, -1]))
        self.light_array.set_light_color(1, tm.vec3([0, 1, 1]))
        self.light_array.set_light_direction(1, tm.vec3([0, -1, -1]))
        self.light_array.set_light_color(2, tm.vec3([1, 0, 1]))
        self.light_array.set_light_direction(2, tm.vec3([-1, 0, -1]))

        self.phong_shader = tdr.BlinnPhongShader(
            camera=self.camera,
            directional_light_array=self.light_array,
            triangle_id_buffer=self.rasterizer.triangle_id_buffer,
            world_position_buffer=self.world_position_interpolator.attribute_buffer,
            normals_buffer=self.normals_interpolator.attribute_buffer,
            albedo_buffer=self.albedo_interpolator.attribute_buffer,
            ambient_constant=0.1,
            specular_constant=0.2,
            diffuse_constant=0.3,
            shine_coefficient=30,
        )

        self.output_buffer = ti.Vector.field(n=3, dtype=float, shape=resolution)
        self.active_layer = ti.field(int, shape=())

    def forward(self):
        self.camera.forward()
        self.rasterizer.forward()
        self.world_position_interpolator.forward()
        self.normals_estimator.forward()
        self.normals_interpolator.forward()
        self.albedo_interpolator.forward()
        self.phong_shader.forward()
        self.copy_output_buffer()

    @ti.kernel
    def copy_output_buffer(self):
        l = self.active_layer[None]
        for x, y in self.output_buffer:
            self.output_buffer[x, y] = self.phong_shader.output_buffer[x, y, l]


if __name__ == "__main__":
    main()
