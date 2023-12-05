from models.Advanced_Network_Helpers import *
from PIL import Image

import numpy as np
import torch as th
import yaml


def read_yaml(path):
    with open(path, "r") as stream:
        try:
            yaml_file = yaml.safe_load(stream)
            print("Yaml file read successfully ✅✅✅")
            return yaml_file
        except yaml.YAMLError as exc:
            print(exc)


def tensors_to_pil(tensors):
    images = (tensors / 2 + 0.5).clamp(0, 1)
    image = images.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid
