import numpy as np
from matplotlib.image import imsave
from matplotlib import pyplot as plt


def plot_image(
    image: np.array, colorbar=False, title=False, cmap="viridis", save_file=None, vmin=None, vmax=None
):
    plt.figure()
    plt.imshow(
        np.swapaxes(image, 0, 1), origin="lower", cmap=cmap, interpolation="nearest", vmax=vmax, vmin=vmin
    )
    plt.tick_params(
        left=False, right=False, labelleft=False, labelbottom=False, bottom=False
    )

    if colorbar:
        plt.colorbar()
    if title:
        plt.title(title)

    plt.tight_layout()

    if save_file != None:
        plt.savefig(save_file, dpi=500)


def show_plots():
    plt.show()


def save_image(image: np.array, file_name: str, vmin=None, vmax=None):
    print()
    print("saving image ", file_name)
    print("image max", np.max(image))
    print("image min", np.min(image))
    imsave(file_name, np.flip(np.swapaxes(image, 0, 1), 0), vmax=vmax, vmin=vmin)