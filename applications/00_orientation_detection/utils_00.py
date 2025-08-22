from typing import Callable
from pathlib import Path
import matplotlib.pyplot as plt
import os
import json
import pprint

from src.radon_transform_algorithm.utils.process_functions import process_images


def process_data(
        input_folder: str, process_function: Callable, process_key: str, display_mode: str,
        normalize: bool, save: bool):
    """
    Select the process function for the different process key
    """
    input_folder = Path(input_folder)

    if process_key == "display":

        process_display_mode_single(input_folder=input_folder, process_function=process_function,
                                    display_mode=display_mode, save=save)

    else:
        process_images(input_path=input_folder, process_function=process_function, process_key=process_key,
                       normalize=normalize, save=save)


def process_display_mode_single(
        input_folder: Path,
        process_function: Callable,
        display_mode: str,
        save: bool
):
    """
    Handles display mode: loads config and result files, then processes images accordingly.

    Parameters
    ----------
    input_folder : Path
        Root folder containing subfolders with config and result JSON files.
    process_function : Callable
        Function to handle displaying results (takes result_data dict as input).
    display_mode : str
        Display mode (used in naming output files if saving).
    save : bool
        Whether to save the resulting plots.
    """
    input_folder = Path(input_folder)

    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_folder}")

    for root, _, _ in os.walk(input_folder):
        root_path = Path(root)

        config_files = sorted(root_path.glob("*config*.json"))
        result_files = sorted(root_path.glob("*results*.json"))

        if not config_files or not result_files:
            print(f"Skipping {root_path}: Missing config or results JSON file.")
            continue

        # Safer: zip longest so mismatches are reported
        for config_file, result_file in zip(config_files, result_files):
            print(f"\nProcessing → config: {config_file.name}, result: {result_file.name}")

            try:
                with config_file.open("r", encoding="utf-8") as f:
                    config_data = json.load(f)

                with result_file.open("r", encoding="utf-8") as f:
                    result_data = json.load(f)

            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse JSON in {config_file} or {result_file}: {e}")
                continue

            # Merge config data into results for display
            result_data.update({
                "input_data_name": config_data.get("input_data_name", "unknown"),
                "input_data_path": config_data.get("input_data_path"),
                "analysis_name": config_data.get("analysis_name"),
            })

            # Display for debugging
            print("📄 Config summary:")
            pprint.pprint(config_data)

            # Call process function
            try:
                process_function(result_data)
            except Exception as e:
                print(f"❌ Error while processing {result_file.name}: {e}")
                continue

            if save:
                data_name = Path(result_data["input_data_name"]).stem
                img_name = f"{display_mode}_{data_name}.png"
                saving_path = root_path / img_name  # save inside same folder
                plt.savefig(saving_path, bbox_inches="tight", dpi=300)
                print(f"✅ Saved plot → {saving_path}")

            plt.show()
            plt.close()
