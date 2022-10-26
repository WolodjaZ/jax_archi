from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision


def visualize_img(imgs: List[np.ndarray]) -> None:
    """Visualize images

    Args:
        imgs (List[np.ndarray]): images to visualize
    """
    imgs = torch.from_numpy(np.array(imgs)).permute(0, 3, 1, 2)
    img_grid = torchvision.utils.make_grid(imgs, nrow=4, normalize=True, pad_value=0.5)
    img_grid = img_grid.permute(1, 2, 0)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(img_grid)
    plt.axis("off")
    plt.show()