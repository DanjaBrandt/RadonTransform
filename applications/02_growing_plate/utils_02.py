from typing import Callable
from pathlib import Path
import matplotlib.pyplot as plt
import os
import json
import pprint

from src.radon_transform_algorithm.utils.process_functions import process_images, import_all_results


def process_data(
        input_folder: str, masks_path: str, process_function: Callable, process_key: str, display_mode: str,
        normalize: bool, save: bool):
    """
    Select the process function for the different process key
    """
    input_folder = Path(input_folder)

    if process_key == "display":

        process_display_mode_all_data(input_folder=input_folder, masks_folder=masks_path, process_function=process_function,
                                      display_mode=display_mode, save=save)

    else:
        process_images(input_path=input_folder, process_function=process_function, process_key=process_key,
                       normalize=normalize, save=save)


def process_display_mode_all_data(input_folder, masks_folder, process_function, display_mode, save):

    results_structure = import_all_results(input_folder)
    #pprint.pprint(results_structure)
    print('masks_folder', masks_folder)
    masks_path = collect_mask_paths(masks_folder)
    pprint.pprint(masks_path)

    process_function(results_structure, masks_path)
    plt.show()

def collect_mask_paths(path: Path) -> dict:
    """
    Recursively builds a nested dict from folders.
    At the leaf level, stores paths to all mask_*.tif files.
    """
    path = Path(path)
    result = {}

    # Leaf: contains mask images
    mask_files = list(path.glob("*_mask.png"))
    if mask_files:
        # Store the Path objects (or str if you prefer)
        return mask_files[0]

    # Otherwise recurse into subfolders
    for item in path.iterdir():
        if item.is_dir():
            result[item.name] = collect_mask_paths(item)

    return result