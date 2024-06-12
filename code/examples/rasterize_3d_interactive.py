from random import random
import taichi as ti

ti.init(arch=ti.cuda)

import tinydiffrast as tdr


def main():
    pipeline = Pipeline(
        resolution=(1024, 1024),
        mesh_type="dragon",
        cam_type="Perspective",
        subpixel_bits=8,
    )

    window = ti.ui.Window("Interactive Rasterizer", res=pipeline.output_buffer.shape)
    canvas = window.get_canvas()
    controller = tdr.CameraController(pipeline.camera)

    print()
    print("*** Running Interactive Rasterizer ***")
    print()
    print("'wasd' keys control camera xy translation")
    print("'eq' keys control camera z translation")
    print("Arrow keys control camera pitch and yaw")
    print("'zx' keys control camera roll")
    print()
    print("'uiop' keys control the visible buffer")
    print("'kl' keys control the visible layer")
    print()

    while window.running:
        pipeline.forward()
        canvas.set_image(pipeline.output_buffer)
        window.show()

        controller.track_inputs(window)
        if window.get_event(ti.ui.PRESS):
            match window.event.key:
                case "u":
                    pipeline.active_buffer = "sil"
                    print("Showing Silhouette Buffer")
                case "i":
                    pipeline.active_buffer = "tri"
                    print("Showing Triangle ID Buffer")
                case "o":
                    pipeline.active_buffer = "z"
                    print("Showing Z Buffer")
                case "p":
                    pipeline.active_buffer = "bary"
                    print("Showing Barycentric Buffer")

                case "k":
                    pipeline.active_layer[None] = 0
                    print("Showing Layer 0")
                case "l":
                    pipeline.active_layer[None] = 1
                    print("Showing Layer 1")


@ti.data_oriented
class Pipeline:

    def __init__(
        self,
        resolution=(1024, 1024),
        mesh_type="dragon",
        cam_type="Perspective",
        subpixel_bits=8,
    ):
        self.mesh = tdr.TriangleMesh.get_example_mesh(mesh_type)

        if cam_type == "Perspective":
            self.camera = tdr.PerspectiveCamera()
        elif cam_type == "orthographic":
            self.camera = tdr.OrthographicCamera()
        else:
            raise Exception(
                "Only Perspective and orthographic cameras are currently supported"
            )

        self.rasterizer = tdr.TriangleRasterizer(
            resolution, self.camera, self.mesh, subpixel_bits=subpixel_bits
        )
        self.output_buffer = ti.Vector.field(n=3, dtype=float, shape=resolution)
        self.active_buffer = "sil"
        self.active_layer = ti.field(int, shape=())

        # random numbers used in triangle id shader
        self.r = random()
        self.g = random()
        self.b = random()

    def forward(self):
        self.camera.forward()
        self.rasterizer.forward()

        # populate output buffer for active layer and raster buffer
        match self.active_buffer:
            case "sil":
                self.shade_silhouette_buffer()
            case "tri":
                self.shade_triangle_buffer()
            case "z":
                self.shade_depth_buffer()
            case "bary":
                self.shade_barycentric_buffer()

    @ti.kernel
    def shade_silhouette_buffer(self):
        l = self.active_layer[None]
        for x, y in self.output_buffer:
            if self.rasterizer.triangle_id_buffer[x, y, l] == 0:
                self.output_buffer[x, y] = 0, 0, 0
            else:
                self.output_buffer[x, y] = 1, 1, 1

    @ti.kernel
    def shade_triangle_buffer(self):
        l = self.active_layer[None]
        for x, y in self.output_buffer:
            tri_id = self.rasterizer.triangle_id_buffer[x, y, l]
            r = tri_id * self.r % 1
            g = tri_id * self.g % 1
            b = tri_id * self.b % 1
            self.output_buffer[x, y] = r, g, b

    @ti.kernel
    def shade_barycentric_buffer(self):
        l = self.active_layer[None]
        for x, y in self.output_buffer:
            if self.rasterizer.triangle_id_buffer[x, y, l] == 0:
                self.output_buffer[x, y] = 0
            else:
                barys = self.rasterizer.barycentric_buffer[x, y, l]
                r = barys.r
                g = barys.g
                b = 1.0 - r - g
                self.output_buffer[x, y] = r, g, b

    @ti.kernel
    def shade_depth_buffer(self):
        l = self.active_layer[None]
        for x, y in self.output_buffer:
            self.output_buffer[x, y] = self.rasterizer.depth_buffer[x, y, l]


if __name__ == "__main__":
    main()
