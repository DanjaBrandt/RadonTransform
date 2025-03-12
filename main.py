
import argparse

from config import Config  # Import the Config class
from radon_analysis import RadonStructureDetection

class MainClass:
    def __init__(self, config: Config, image_path: str):
        self.config = config
        self.processor = RadonStructureDetection(config, image_path)

    def run(self):
        print(f"Running with config: {self.config}")
        self.processor.generate()


def get_args():
    parser = argparse.ArgumentParser(description="Provide config values")
    parser.add_argument("--patch_size", type=int, default=50, help="Set the patch size")
    parser.add_argument("--patch_step", type=str, default=25, help="Set the step size for patches")
    parser.add_argument("--sigma", type=str, default=1, help="Set the sigma value for Gauss smoothing")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the image folder")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    config = Config(**vars(args))
    app = MainClass(config, args.image_path)
    app.run()

