
class RadonStructureDetection:
    def __init__(self, config, image_path: str):
        self.config = config

    def apply_gaussian_filter(self, image):
        """Apply Gaussian blur to the given image using sigma from the config."""
        sigma = self.config.sigma
        return cv2.GaussianBlur(image, (0, 0), sigma)

    def process(self, image):
        """Main processing function (e.g., apply filtering)."""
        print(f"Processing image with sigma={self.config.sigma}")
        return self.apply_gaussian_filter(image)