import os
import argparse
import sys
from pathlib import Path

from config import Config  # Import the Config class
from radon_analysis import RadonStructureDetection
from display_handler import DisplayHandler
from coregistered_display import DisplayCoregisterdImages
from logger_config import setup_logger
from utils.process_functions import process_images_in_folder
from utils import functions


class MainClass:
    """
        Main class to handle different processing types (Radon, Fourier, Display).
        """

    def __init__(self, config, process_key, display_mode=None, acquisition_mode=None, normalize=True, save=True):
        self.config = config
        self.process_key = process_key
        self.display_mode = display_mode
        self.acquisition_mode = acquisition_mode
        self.normalize = normalize
        self.save = save

        if process_key == "display":
            self.processor = DisplayHandler(display_mode).display
        elif process_key == "coregistered_display":
            self.processor = DisplayCoregisterdImages(display_mode, acquisition_mode).display
        elif process_key == "radon":
            self.processor = RadonStructureDetection(config).process
        # elif process_key == "fourier":
        # self.processor = FourierTransform.process
        else:
            # self.logger.error(f"Invalid process key: {process_key}")
            raise ValueError(f"Unknown process_key: {process_key}")

    def run(self, input_folder):

        process_images_in_folder(input_folder, self.processor, self.process_key, self.display_mode,
                                 self.acquisition_mode, self.normalize, self.save)


def get_args():
    parser = argparse.ArgumentParser(description="Provide config values")

    parser.add_argument("--mode", type=str, choices=["radon", "fourier", "display"], required=True,
                        help="Choose processing type: 'radon', 'fourier', or 'display'")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the input folder")
    parser.add_argument("--display_mode", type=str, default="image", choices=["angle_differences",
                                                                              "local_alignment",
                                                                              "{fit_|ratio_}angle_differences_all_alldays",
                                                                              "{fit_|ratio_}angle_differences_all_{single day}",
                                                                              "polar_histogram_all",
                                                                              "count_points_all"
                                                                              ],
                        # if angle_differences and local alignment the path must contain the acquisition mode
                        # (or 2PM or SHG) to be able
                        # to compair. It can compute the hist for all result, or give a specific result folder.
                        # example folder: "./outputs/radon_output_02/2PM"
                        # if all the input folder should be before the folder of the acquisition modes
                        # example folder: "./outputs/radon_output_02"
                        help="Choose display mode (for 'display' mode only)")
    parser.add_argument("--acquisition_mode", type=list, default=["2PM", "SHG"],
                        help="If cooregistered images, the outer folders are the aquisition modalities")
    parser.add_argument("--save", type=str, default="True", choices=["True", "False"],
                        help="Choose if to save results")

    parser.add_argument("--patch_size", type=int, default=50, help="Set the patch size")
    parser.add_argument("--patch_step", type=str, default=25, help="Set the step size for patches")
    parser.add_argument("--sigma", type=str, default=1, help="Set the sigma value for Gauss smoothing")
    parser.add_argument("--normalize", type=bool, default=False, help="Normalize the image")

    return parser.parse_args()


if __name__ == "__main__":
    if len(sys.argv) > 1:  # If there are command-line arguments
        args = get_args()
    else:  # No CLI arguments, use default values
        args = argparse.Namespace(
            mode="radon",
            #input_folder="./outputs/radon_output_02",
            #mode="radon",
            #input_folder="data_test/SHG/m17_ll/day_14",
            input_folder="bon/SHG",
            display_mode="ratio_angle_differences_all_14", #"angle_differences_all, angle_differences, local_alignment",
            acquisition_mode=["2PM", "SHG"],
            patch_size=500,
            patch_step=250,
            sigma=1,
            normalize=True,
            save=False
        )

    input_config = Config(**vars(args))
    # app = MainClass(config, args.image_path)
    # app.run()
    app = MainClass(input_config, args.mode, args.display_mode, args.acquisition_mode, args.normalize, args.save)
    app.run(args.input_folder)

    # logger.info("Application finished")
