from typing import Callable
from pathlib import Path
import matplotlib.pyplot as plt
import os
import json
import pprint

from src.radon_transform_algorithm.utils.process_functions import process_images, import_all_results


def process_data(
        input_folder: str, process_function: Callable, process_key: str, display_mode: str,
        normalize: bool, save: bool):
    """
    Select the process function for the different process key
    """
    input_folder = Path(input_folder)

    if process_key == "display":

        process_display_mode_all_data(input_folder=input_folder, process_function=process_function,
                                      display_mode=display_mode, save=save)

    else:
        process_images(input_path=input_folder, process_function=process_function, process_key=process_key,
                       normalize=normalize, save=save)


def process_display_mode_all_data(input_folder, process_function, display_mode, save):

    results_structure = import_all_results(input_folder)
    #pprint.pprint(results_structure)

    results = {}
    for group, mice in results_structure.items():
        #print(group)
        #for mouse, days in mice.items():
            #print(mouse)
        num_mice = len(mice)
        days_per_mouse = {mouse: len(days) for mouse, days in mice.items()}
        results[group] = {
            "total_mice": num_mice,
            "days_per_mouse": days_per_mouse
        }
    #pprint.pprint(results)
    process_function(results_structure)


