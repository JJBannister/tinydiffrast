import taichi as ti
import taichi.math as tm
import matplotlib.pyplot as plt
import numpy as np

ti.init(arch=ti.gpu)


# Define Data Containers
triangle = ti.Vector.field(dtype=ti.f32, n=2, shape=(3), needs_grad=True)
triangle[0] = tm.vec2(10.1, 10.2)
triangle[1] = tm.vec2(90.3, 10.4)
triangle[2] = tm.vec2(50.5, 50.6)

image = ti.field(dtype=ti.f32, shape=(100, 100), needs_grad=True)
loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)


# Define Kernels and Device Functions
@ti.kernel
def render():
    for x, y in image:
        if is_pixel_in_triangle(tm.vec2([x, y])):
            image[x, y] = 1.0
        else:
            image[x, y] = 0.0


@ti.func
def is_pixel_in_triangle(pixel: tm.vec2) -> bool:
    return (
        is_edge_function_positive(pixel, triangle[1], triangle[0])
        and is_edge_function_positive(pixel, triangle[2], triangle[1])
        and is_edge_function_positive(pixel, triangle[0], triangle[2])
    )


@ti.func
def is_edge_function_positive(pixel: tm.vec2, p0: tm.vec2, p1: tm.vec2):
    return tm.cross(pixel - p0, p1 - p0) >= 0


@ti.kernel
def compute_loss():
    for x, y in image:
        loss[None] += image[x, y]


def show_image(np_array, title: str = None):
    plt.figure()
    plt.imshow(np.swapaxes(np_array, 0, 1), origin="lower", vmin=0, vmax=1)
    plt.tick_params(
        left=False, right=False, labelleft=False, labelbottom=False, bottom=False
    )
    plt.colorbar()
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.show()


# Run Experiments

# Forward Rendering
render()
compute_loss()
print("Loss = ", loss[None])
print()
show_image(image.to_numpy(), title="A Basic Triangle")

# Compute grad of loss kernel
loss.grad[None] = 1.0
compute_loss.grad()
show_image(image.grad.to_numpy(), title="Image Gradient")

# Compute grad of render kernel
render.grad()
print("Triangle Grad")
print(triangle.grad.to_numpy())
print()

# Compute Finite Difference
loss.fill(0.0)

triangle[2].y += 1e-5
render()
compute_loss()
print("Perturbed Loss = ", loss[None])
print()
show_image(image.to_numpy(), title="A Perturbed Triangle")

# Investigate Loss Landscape
loss_sequence = []
for i in range(10000):
    loss.fill(0.0)
    triangle[2].y += 1e-5
    render()
    compute_loss()
    loss_sequence.append(loss[None])

plt.plot([i * 1e-5 for i in range(10000)], loss_sequence)
plt.title("Loss vs. Perturbation Size", fontsize=16)
plt.ylabel("Loss", fontsize=14)
plt.xlabel("Perturbation Size (pixels)", fontsize=14)
plt.tight_layout()
plt.show()
