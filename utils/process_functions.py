import os

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Callable
from pathlib import Path
import json
import pprint

#from logger_config import get_logger
from utils import functions

def import_data(path: str, normalize: bool = True, max_value: int = 255, min_value: int = 0) -> np.ndarray:
    """
    Opens an image or a stack and returns the bits of the image,
    the array of the image and the shape of the array
    :param path: Path to the file to be read
    :type path: str
    :param normalize: if True outputs image normalized between max_value and min_value
    :type normalize: bool
    :param max_value: max value of grayscale values
    :type max_value: int
    :param min_value: min value of grayscale values
    :type min_value: int
    :return: the image stack (if just 1 img shape:[1,:,:,:])
    :rtype: np.ndarray
    """

    try:
        # Read PIL image
        with Image.open(path) as img:
            # Read image depth
            mode = img.mode
            '''
            if mode.startswith('L'):
                logger.info(f"{mode}: 8-bit grayscale imported data")
            elif mode.startswith('I;16'):
                logger.info(f"{mode}: 16-bit grayscale imported data")
            elif mode.startswith('RGB'):
                logger.info(f"{mode}: 24-bit color imported data")
                img = img.convert('L')
                logger.info("Imported RGB image converted to 8-bit grayscale image")
            else:
                logger.info(f"{mode}: Unknown image depth for PIL type")'''

            # Create image array
            h, w = img.size
            imarray = np.array(img)
            tiffarray = np.zeros((img.n_frames, h, w))
            for i in range(img.n_frames):
                img.seek(i)
                # Normalize the image if normalized=True
                if normalize:
                    #logger.info(f'Image is normalized between {min_value} and {max_value}')
                    norm = ((imarray - np.min(imarray)) * (
                            (max_value - min_value) / (np.max(imarray) - np.min(imarray))) + min_value)
                    tiffarray[i, :, :] = norm
                else:
                    #logger.info('Attention! Image is not normalized')
                    tiffarray[i, :, :] = imarray

            #logger.info(f'Imported data of shape: {tiffarray.shape}. Data-type: {tiffarray[0][0][0].dtype}')

    except FileNotFoundError:
        #logger.error("Data file not found")
        print(("Data file not found"))

    return tiffarray


def process_images_in_folder(
    input_folder: str, process_function: Callable, process_key: str, normalize: bool = True
):
    """
    Processes all images in a given folder and saves the results.

    :param input_folder: The root folder containing input images.
    :param process_function: A function that processes a NumPy array and returns an output.
    :param process_key: The name of the process function (Radon, Fourier...)
    :param normalize: Whether to normalize the images.
    """

    input_folder = Path(input_folder)

    if process_key == "display":
        process_display_mode(input_folder, process_function, normalize)
    else:
        process_and_save_images(input_folder, process_function, process_key, normalize)


def process_display_mode(input_folder: Path, process_function: Callable, normalize: bool):
    """Handles display mode: loads config and result files, then processes images accordingly."""

    for root, _, _ in os.walk(input_folder):
        root_path = Path(root)

        config_file = next(root_path.glob("*config*.json"), None)
        result_file = next(root_path.glob("*results*.json"), None)

        if not config_file or not result_file:
            print(f"Skipping {root_path}: Missing config or results JSON file.")
            continue  # Skip folders without required files

        # Load config data
        with config_file.open("r", encoding="utf-8") as f:
            config_data = json.load(f)

        # Load results data
        with result_file.open("r", encoding="utf-8") as f:
            result_data = json.load(f)

        # Display results
        print("\nDisplaying results for config:\n")
        pprint.pprint(config_data)

        result_data.update(
            {
                "input_data_name": config_data.get("input_data_name"),
                "input_data_path": config_data.get("input_data_path"),
                "analysis_name": config_data.get("analysis_name"),
            }
        )

        # Call process function with required parameters
        process_function(result_data)


def process_and_save_images(input_folder: Path, process_function: Callable, process_key: str, normalize: bool):
    """Processes and saves images in a structured output folder."""

    output_root = Path("outputs")
    base_output_folder = f"{process_key}_output"
    output_folder = functions.get_unique_name(base_name=base_output_folder, parent_dir=output_root)

    for root, _, files in os.walk(input_folder):
        root_path = Path(root)

        for file in files:
            file_path = root_path / file
            image_array = import_data(str(file_path), normalize=normalize)
            if image_array is None:
                continue  # Skip invalid images

            relative_path = root_path.relative_to(input_folder)
            output_dir = output_folder / relative_path
            output_dir.mkdir(parents=True, exist_ok=True)

            config, detected_points = process_function(image_array)
            config.update(
                {
                    "input_data_name": file,
                    "input_data_path": str(file_path),
                    "analysis_name": process_key,
                }
            )

            results_name = f"results_{file_path.stem}"
            config_name = f"config_{file_path.stem}"

            functions.write_results(results_name, detected_points, str(output_dir))
            functions.write_config(config_name, config, str(output_dir))


def get_unique_output_folder(base_name: str) -> str:
    """
    Checks if the output folder already exists and finds a unique name by appending numbers (_01, _02, ...).

    :param base_name: The base name of the folder (e.g., "radon_output")
    :return: A unique folder name (e.g., "radon_output_01" or "radon_output_02")
    """
    if not os.path.exists(base_name):
        return base_name  # If it doesn't exist, use the base name

    counter = 1
    while True:
        new_folder = f"{base_name}_{counter:02d}"
        if not os.path.exists(new_folder):
            return new_folder  # Return the first available unique name
        counter += 1