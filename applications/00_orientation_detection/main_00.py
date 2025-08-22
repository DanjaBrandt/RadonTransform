import argparse
import sys

from src.config import Config
from src.radon_transform_algorithm.radon_analysis import RadonStructureDetection

from utils_00 import process_data

from display_handler_00 import DisplayHandler


class MainClass:
    """
        Main class to handle the RadonAlgorithm analysis and display the results.

        Args:
            config (dict): Configuration dictionary
            process_key (str): Key to determine the process type
            **kwargs: Optional arguments:
                - input_folder (Path or str)
                - result_folder (Path or str, default=None)
                - display_mode (str, default=None)
                - normalize (bool, default=True)
                - save (bool, default=True)
        """

    def __init__(self, config, process_key, **kwargs):
        self.config = config
        self.process_key = process_key

        self.input_folder = kwargs.get("input_folder")
        self.result_folder = kwargs.get("result_folder", None)
        self.display_mode = kwargs.get("display_mode", None)
        self.normalize = kwargs.get("normalize", True)
        self.save = kwargs.get("save", True)

        if process_key == "display":
            self.processor = DisplayHandler(self.display_mode).display
            self.process_folder = self.result_folder
        elif process_key == "radon":
            self.processor = RadonStructureDetection(config).process
            self.process_folder = self.input_folder
        else:
            raise ValueError(f"Unknown process_key: {process_key}")

    def run(self):
        """
                Run the selected processor on images in the given folder.
                """
        process_data(input_folder=self.process_folder, process_function=self.processor, process_key=self.process_key,
                     display_mode=self.display_mode, normalize=self.normalize, save=self.save)


def get_args():
    """
        Parse command line arguments.
        """

    parser = argparse.ArgumentParser(description="Image processing pipeline")

    parser.add_argument(
        "--mode",
        type=str,
        choices=["radon", "display"],
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
            "image",
            "polar_histogram",
            "widths",
        ],
        help="Choose display mode (for 'display' only)."
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
            mode="radon",
            input_folder="00_data/real",
            result_folder="00_outputs/00_radon_output_00/real/2PM",
            display_mode="image",
            patch_size=100,
            patch_step=100,
            sigma=1.0,
            normalize=True,
            save=0
        )

    # Build config and run
    input_config = Config(**vars(args))
    app = MainClass(
        config=input_config,
        process_key=args.mode,
        input_folder=args.input_folder,
        result_folder=args.result_folder,
        display_mode=args.display_mode,
        normalize=args.normalize,
        save=args.save
    )
    app.run()
    print("Application finished successfully.")
