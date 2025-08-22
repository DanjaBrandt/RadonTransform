import argparse
import sys

from src.config import Config
from src.radon_transform_algorithm.radon_analysis import RadonStructureDetection

from utils_02 import process_data

from display_handler_02 import DisplayGrowingPlate


class MainClass:
    """
        Main class to handle the analysis of coregisterd data (Fluorescence and SHG)
        using the Radon Algorithm and display the results.

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
        self.masks_folder = kwargs.get("masks_folder")
        self.result_folder = kwargs.get("result_folder", None)
        self.display_mode = kwargs.get("display_mode", None)
        self.normalize = kwargs.get("normalize", True)
        self.save = kwargs.get("save", True)

        if process_key == "display":
            self.processor = DisplayGrowingPlate(**kwargs).display
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
        process_data(input_folder=self.process_folder, masks_path=self.masks_folder, process_function=self.processor, process_key=self.process_key,
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
            "images",
            "local_alignment",
            "angle_differences",
            #"widths",
            #"widths_over_days"
        ],
        help="Choose display mode (for 'display' or 'coregistered_display' only)."
    )

    parser.add_argument(
        "--mouse",
        type=str,
        default=None,
        help="Choose one mice to display. If None all mice will be displayed."
    )

    parser.add_argument(
        "--day",
        type=str,
        default=None,
        help="Choose one day to display. If None all days will be displayed."
    )

    parser.add_argument(
        "--coregistered_channels",
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
            input_folder="02_data",
            masks_folder="02_masks",
            result_folder="02_outputs/02_radon_output",
            display_mode="angle_alignment",
            group='middle',
            mouse='m1',
            masked_region=None,
            patch_size=1000,
            patch_step=500,
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
        masks_folder=args.masks_folder,
        result_folder=args.result_folder,
        display_mode=args.display_mode,
        group=args.group,
        mouse=args.mouse,
        masked_region=args.masked_region,
        normalize=args.normalize,
        save=args.save
    )
    app.run()
    print("Application finished successfully.")
