import numpy as np


def create_circular_mask(height: int, width: int, center: tuple = None, radius: int = None) -> np.ndarray:
    """
     Creates a circular mask
    :param height: Height of the mask
    :param width: Width of the mask
    :param center: Center of the circle. Defaults to the middle of the image
    :param radius: Radius of the circle. Defaults to the smallest distance between
                               the center and image borders
    :return: Circular mask
    """

    if center is None:
        center = (width // 2, height // 2)

    if radius is None:
        radius = min(center[0], center[1], width - center[0], height - center[1])

    y, x = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

    mask = dist_from_center <= radius

    return mask
