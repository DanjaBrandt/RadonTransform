import argparse
import sys

from config import Config
from src.radon_transform_algorithm.radon_analysis import RadonStructureDetection
from display_handler import DisplayHandler
from coregistered_display import DisplayCoregisterdImages
from src.radon_transform_algorithm.utils.process_functions import process_images_in_folder


class MainClass:
    """
        Main class to handle different processing types (Radon, Display, Coregistered Display).
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
        else:
            raise ValueError(f"Unknown process_key: {process_key}")

    def run(self, input_folder):
        """
                Run the selected processor on images in the given folder.
                """
        process_images_in_folder(input_folder, self.processor, self.process_key, self.display_mode,
                                 self.acquisition_mode, self.normalize, self.save)


def get_args():
    """
        Parse command line arguments.
        """

    parser = argparse.ArgumentParser(description="Image processing pipeline")

    parser.add_argument(
        "--mode",
        type=str,
        choices=["radon", "display", "coregistered_display"],
        required=True,
        help="Choose processing type."
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Path to the input folder containing images."
    )
    parser.add_argument(
        "--display_mode",
        type=str,
        default="image",
        choices=[
            "image", #display
            "polar_histogram", #display
            "polar_histogram_over_days", #display
            "widths", #display
            "widths_over_days", #display
            "angle_differences",
            "local_alignment",
            "fit_angle_differences_all_alldays",
            "ratio_angle_differences_all_alldays",
            "fit_angle_differences_all_single_day",
            "ratio_angle_differences_all_single_day",
            "polar_histogram_all",
            "count_points_all"
        ],
        help="Choose display mode (for 'display' or 'coregistered_display' only)."
    )
    parser.add_argument(
        "--acquisition_mode",
        nargs="+",
        default=["2PM", "SHG"],
        help="List of acquisition modalities (e.g., 2PM SHG)."
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="If set, save the results to disk."
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="If set, normalize the input images."
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=50,
        help="Set the patch size."
    )
    parser.add_argument(
        "--patch_step",
        type=int,
        default=25,
        help="Set the step size for patches."
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="Set the sigma value for Gaussian smoothing."
    )

    return parser.parse_args()


if __name__ == "__main__":
    if len(sys.argv) > 1:  # CLI mode
        args = get_args()
    else:  # Default mode (useful for testing / notebooks)
        args = argparse.Namespace(
            mode="display",
            input_folder="outputs/00_radon_output_00/real/2PM",
            display_mode="polar_histogram",
            acquisition_mode=["2PM", "SHG"],
            patch_size=500,
            patch_step=250,
            sigma=1.0,
            normalize=True,
            save=False
        )

    # Build config and run
    input_config = Config(**vars(args))
    app = MainClass(
        input_config,
        args.mode,
        args.display_mode,
        args.acquisition_mode,
        args.normalize,
        args.save
    )
    app.run(args.input_folder)
    print("Application finished successfully.")