from typing import Any, Tuple

import numpy as np
from PIL import Image

DATASETS = {
    "CIFAR10": [
        [0.4914, 0.4822, 0.4465],
        [0.2470, 0.2435, 0.2616]
    ]
}


def image_to_numpy(img: Image.Image, dataset_name: str) -> np.ndarray:
    """Transformations applied on each image => bring them into a numpy array

    Args:
        img (Image.Image): original image
        dataset_name (str): name of the dataset for mean and var

    Returns:
        np.ndarray: normalized numpy array image
    """
    if dataset_name not in DATASETS:
        raise ValueError(f"Dataset {dataset_name} not found in {DATASETS.keys()}")
    
    dataset = DATASETS[dataset_name]
    img = np.array(img, dtype=np.float32)
    img = (img / 255. - dataset[0]) / dataset[1]
    return img


def numpy_collate(batch: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Stack the batch elements

    Args:
        batch (Any): batch of elements

    Returns:
        Tuple[np.ndarray, np.ndarray]: stacked batch
    """
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)