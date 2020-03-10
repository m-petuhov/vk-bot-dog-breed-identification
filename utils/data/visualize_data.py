import numpy as np

from config import cfg


def image_show(tensor_image, plt, title=None):
    image = tensor_image.numpy().transpose([1, 2, 0])
    image = np.array(cfg.transform_params['std']) * image + np.array(cfg.transform_params['mean'])
    image = np.clip(image, 0, 1)

    plt.imshow(image)

    if title is not None:
        plt.set_title(title)
    plt.grid(False)
