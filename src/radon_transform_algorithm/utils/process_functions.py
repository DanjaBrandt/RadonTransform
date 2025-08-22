import os

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Callable
from pathlib import Path
import json
import pprint
import re

from . import functions


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

            if mode.startswith('RGB'):
                print(f"{mode}: 24-bit color imported data")
                img = img.convert('L')
                print("Imported RGB image converted to 8-bit grayscale image")

            # Create image array
            w, h = img.size
            imarray = np.array(img)
            n_frames = getattr(img, 'n_frames', 1)
            tiffarray = np.zeros((n_frames, h, w))
            for i in range(n_frames):
                img.seek(i)
                # Normalize the image if normalized=True
                if normalize:
                    # logger.info(f'Image is normalized between {min_value} and {max_value}')
                    norm = ((imarray - np.min(imarray)) * (
                            (max_value - min_value) / (np.max(imarray) - np.min(imarray))) + min_value)
                    tiffarray[i, :, :] = norm
                else:
                    # logger.info('Attention! Image is not normalized')
                    tiffarray[i, :, :] = imarray

            # logger.info(f'Imported data of shape: {tiffarray.shape}. Data-type: {tiffarray[0][0][0].dtype}')

    except FileNotFoundError:
        # logger.error("Data file not found")
        print(("Data file not found"))

    return tiffarray

def process_images(
        input_path: Path,
        process_function: Callable,
        process_key: str,
        normalize: bool,
        save: bool = True
):
    """Processes and saves images in a structured output folder.

    Works with either a directory or a single image file.
    """

    input_path = Path(input_path)

    # Handle prefix: works if input is a file or a folder
    prefix = input_path.parts[0].split("_")[0]

    # Build output folder name
    output_root = Path(f"{prefix}_outputs")
    base_output_folder = f"{prefix}_{process_key}_output"
    output_folder = Path(functions.get_unique_name(base_name=base_output_folder, parent_dir=output_root))

    # Case 1: input is a single file
    if input_path.is_file():
        files = [input_path]
        root = input_path.parent
        walk_iter = [(root, [], [input_path.name])]
    else:
        # Case 2: input is a directory
        walk_iter = os.walk(input_path)

    for root, _, files in walk_iter:
        root_path = Path(root)

        for file in files:
            file_path = root_path / file
            image_array = import_data(str(file_path), normalize=normalize)
            if image_array is None:
                continue  # Skip invalid images

            # Preserve relative structure if input is a folder
            relative_path = root_path.relative_to(input_path if input_path.is_dir() else root_path)
            output_dir = output_folder / relative_path

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

            if save:
                output_dir.mkdir(parents=True, exist_ok=True)
                functions.write_results(results_name, detected_points, str(output_dir))
                functions.write_config(config_name, config, str(output_dir))


def import_all_results(path: Path) -> dict:
    """
    Recursively builds a nested dict from folders.
    At the last level, pairs config_*.json and results_*.json files,
    merges them, and stores the enriched results directly (no tif filename as key).
    """
    path = Path(path)
    result = {}
    # Collect all config_*.json files
    config_files = list(path.glob("config_*.json"))
    # If we have config files, treat this as a leaf folder
    if config_files:
        # Assume only one pair per folder
        config_file = config_files[0]
        suffix = config_file.stem.replace("config_", "")
        result_file = path / f"results_{suffix}.json"
        if result_file.exists():
            # Load config
            with config_file.open("r", encoding="utf-8") as f:
                config_data = json.load(f)
            # Load results
            with result_file.open("r", encoding="utf-8") as f:
                result_data = json.load(f)
            # Merge metadata from config into results
            result_data.update(
                {
                    "input_data_name": config_data.get("input_data_name"),
                    "input_data_path": config_data.get("input_data_path"),
                    "analysis_name": config_data.get("analysis_name"),
                }
            )
            return result_data
    # Otherwise recurse into subfolders
    for item in path.iterdir():
        if item.is_dir():
            result[item.name] = import_all_results(item)
    return result

######################remove#############################
def process_images_in_folder(
        input_folder: str, process_function: Callable, process_key: str, display_mode: str, acquisition_mode: list,
        normalize: bool = True, save: bool = True):
    """
    Processes all images in a given folder and saves the results.

    :param input_folder: The root folder containing input images.
    :param process_function: A function that processes a NumPy array and returns an output.
    :param process_key: The name of the process function (Radon, Fourier...)
    :param normalize: Whether to normalize the images.
    """
    input_folder = Path(input_folder)

    if process_key == "display":
        if "over_days" in display_mode:
            process_display_mode_over_days(input_folder=input_folder, process_function=process_function,
                                           display_mode=display_mode, save=save)
        else:

            process_display_mode(input_folder=input_folder, process_function=process_function,
                                 display_mode=display_mode, save=save)
    elif process_key == "coregistered_display":
        if "all" in display_mode:
            process_all_coregisterd_display(input_folder=input_folder, process_function=process_function,
                                            display_mode=display_mode, acquisition_mode=acquisition_mode, save=save)
        else:
            process_coregisterd_display(input_folder=input_folder, process_function=process_function,
                                        display_mode=display_mode, acquisition_mode=acquisition_mode, save=save)
    else:
        process_and_save_images(input_folder=input_folder, process_function=process_function, process_key=process_key,
                                normalize=normalize, save=save)


def process_display_mode(input_folder: Path, process_function: Callable, display_mode: str, save: bool):
    """Handles display mode: loads config and result files, then processes images accordingly."""
    for root, _, files in os.walk(input_folder):
        root_path = Path(root)

        config_files = list(root_path.glob("*config*.json"))
        result_files = list(root_path.glob("*results*.json"))

        if not config_files or not result_files:
            print(f"Skipping {root_path}: Missing config or results JSON file.")
            continue  # Skip folders without required files

        # Iterate over each pair of config and result files
        for config_file, result_file in zip(config_files, result_files):
            print(f"Processing config: {config_file.name}, result: {result_file.name}")

            # Load config data
            with config_file.open("r", encoding="utf-8") as f:
                config_data = json.load(f)

            # Load results data
            with result_file.open("r", encoding="utf-8") as f:
                result_data = json.load(f)

            data_name = config_data.get("input_data_name")

            result_data.update(
                {
                    "input_data_name": data_name,
                    "input_data_path": config_data.get("input_data_path"),
                    "analysis_name": config_data.get("analysis_name"),
                }
            )

            # Display results
            print("\nDisplaying results for config:\n")
            pprint.pprint(config_data)
            # Call process function with the required parameters
            process_function(result_data)

            if save:
                data_mouse_day = re.match(r'^(.*?-d\d+)', data_name)
                base = data_mouse_day.group(1)
                plt.savefig(f'{base}_{display_mode}.png', bbox_inches='tight')
            plt.show()


def process_display_mode_over_days(input_folder: Path, process_function: Callable, display_mode: str, save: bool):
    """Handles display mode: loads config and result files, then processes images accordingly."""

    result_over_days = {}
    for root, _, files in os.walk(input_folder):
        print('ROOT', root)
        root_path = Path(root)

        config_files = list(root_path.glob("*config*.json"))
        result_files = list(root_path.glob("*results*.json"))

        if not config_files or not result_files:
            print(f"Skipping {root_path}: Missing config or results JSON file.")
            continue  # Skip folders without required files

        # Iterate over each pair of config and result files
        for config_file, result_file in zip(config_files, result_files):
            print(f"Processing config: {config_file.name}, result: {result_file.name}")

            # Load config data
            with config_file.open("r", encoding="utf-8") as f:
                config_data = json.load(f)

            # Load results data
            with result_file.open("r", encoding="utf-8") as f:
                result_data = json.load(f)

            data_name = config_data.get("input_data_name")

            result_data.update(
                {
                    "input_data_name": data_name,
                    "input_data_path": config_data.get("input_data_path"),
                    "analysis_name": config_data.get("analysis_name"),
                }
            )

            day = os.path.basename(root)
            if day not in result_over_days:
                result_over_days[day] = {}

            result_over_days[day] = result_data
        # pprint.pprint(result_over_days)

    process_function(result_over_days)

    if save:
        data_mouse_day = re.match(r'^(.*?)-d\d+', data_name)
        base = data_mouse_day.group(1)
        plt.savefig(f'{base}_{display_mode}.png', bbox_inches='tight')
    plt.show()


def process_coregisterd_display(input_folder: Path, process_function: Callable, display_mode: str,
                                acquisition_mode: list,
                                save: bool):
    # Find where in the path the acquisition_mode appears
    parts = list(input_folder.parts)
    match_index = next((i for i, part in enumerate(parts) if part in acquisition_mode), None)

    if match_index is None:
        print("No matching option found in input_folder.")
        return  # Exit early if no match found

    matched_option = parts[match_index]

    # Generate all paths (original + substituted)
    all_paths = [input_folder]  # Start with the original path
    print('all_paths', all_paths)

    for opt in acquisition_mode:
        if opt != matched_option:
            modified_parts = parts.copy()
            modified_parts[match_index] = opt
            all_paths.append(Path(*modified_parts))

    coregistered_result = {}
    for folder in all_paths:
        mode = [key for key in acquisition_mode if key in str(folder)][0]
        if folder.exists():
            print(f"Processing folder: {folder}")
            for root, _, files in os.walk(folder):
                root_path = Path(root)

                config_files = list(root_path.glob("*config*.json"))
                result_files = list(root_path.glob("*results*.json"))

                if not config_files or not result_files:
                    print(f"Skipping {root_path}: Missing config or results JSON file.")
                    continue

                # Iterate over each pair of config and result files
                for config_file, result_file in zip(config_files, result_files):
                    print(f"Processing config: {config_file.name}, result: {result_file.name}")
                    # Load config and result data
                    result_data = load_and_update_data(config_file, result_file)
                    if result_data:
                        coregistered_result[mode] = result_data
        else:
            print(f"Skipped: {folder} does not exist.")

    # pprint.pprint(coregistered_result)
    process_function(coregistered_result)
    plt.show()


def process_all_coregisterd_display(input_folder: Path, process_function: Callable, display_mode: str,
                                    acquisition_mode: list,
                                    save: bool):
    print(input_folder)
    print(acquisition_mode)
    base = Path(input_folder)
    total_result = {}

    for top_level in base.iterdir():
        if top_level.is_dir():
            total_result[top_level.name] = {}
            for mid_level in top_level.iterdir():
                if mid_level.is_dir():
                    total_result[top_level.name][mid_level.name] = {}

                    for low_level in mid_level.iterdir():
                        if low_level.is_dir():
                            config_files = list(low_level.glob("*config*.json"))
                            result_files = list(low_level.glob("*results*.json"))

                            if not config_files or not result_files:
                                print(f"Skipping {low_level}: Missing config or results JSON file.")
                                continue
                            for config_file, result_file in zip(config_files, result_files):
                                # print(f"Processing config: {config_file.name}, result: {result_file.name}")
                                # Load config and result data
                                result_data = load_and_update_data(config_file, result_file)
                                if result_data:
                                    total_result[top_level.name][mid_level.name][low_level.name] = result_data

    # pprint.pprint(total_result)
    process_function(total_result)
    # plt.show()


def load_and_update_data(config_file: Path, result_file: Path) -> dict:
    """Helper function to load and update result data with config data."""
    try:
        with config_file.open("r", encoding="utf-8") as f:
            config_data = json.load(f)

        with result_file.open("r", encoding="utf-8") as f:
            result_data = json.load(f)

        data_name = config_data.get("input_data_name")

        result_data.update({
            "input_data_name": data_name,
            "input_data_path": config_data.get("input_data_path"),
            "analysis_name": config_data.get("analysis_name"),
        })

        return result_data
    except Exception as e:
        print(f"Error processing files {config_file} and {result_file}: {e}")
        return None


'''
def process_and_save_images(input_folder: Path, process_function: Callable, process_key: str, normalize: bool,
                            save=True):
    """Processes and saves images in a structured output folder."""

    prefix = Path(input_folder.parts[0]).name.split("_")[0]

    # Build output folder name
    output_root = Path(f"{prefix}_outputs")

    #output_root = Path("outputs")
    base_output_folder = f"{prefix}_{process_key}_output"

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

            if save:
                output_dir.mkdir(parents=True, exist_ok=True)
                print(detected_points)
                functions.write_results(results_name, detected_points, str(output_dir))
                functions.write_config(config_name, config, str(output_dir))
'''


def get_unique_output_folder(base_name: str) -> str:
    """
    Checks if the output folder already exists and finds a unique name by appending numbers (_01, _02, ...).

    :param base_name: The base name of the folder (e.g., "radon_output")
    :return: A unique folder name (e.g., "radon_output_01" or "01_radon_output_00")
    """
    if not os.path.exists(base_name):
        return base_name  # If it doesn't exist, use the base name

    counter = 1
    while True:
        new_folder = f"{base_name}_{counter:02d}"
        if not os.path.exists(new_folder):
            return new_folder  # Return the first available unique name
        counter += 1
