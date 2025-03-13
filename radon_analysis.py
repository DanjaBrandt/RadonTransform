import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from utils import functions


class RadonStructureDetection:
    def __init__(self, config):
        self.config = config
        self._background_threshold_value = None

    def apply_gaussian_filter(self, image):
        """Apply Gaussian blur to the given image using sigma from the config."""
        sigma = self.config.sigma
        print(f"sigma {sigma}")
        # return cv2.GaussianBlur(image, (0, 0), sigma)

    def _select_background(self, img_array):
        """Allows user to select a region in an image and computes the mean pixel value."""
        try:
            points = self._get_user_selected_points(img_array)
            if points is None:
                return

            (x1, y1), (x2, y2) = self._validate_coordinates(points)
            selected_region = img_array[y1:y2, x1:x2]
            self._background_threshold_value = np.mean(selected_region)

            print(f"Background pixel value stored: {self._background_threshold_value:.2f}")
        except Exception as e:
            print(f"Error in region selection: {e}")

    @staticmethod
    def _get_user_selected_points(img_array):
        """Displays image and allows user to click two points."""
        plt.imshow(img_array, cmap='gray')
        plt.title("Click two points: Top-Left & Bottom-Right")
        plt.axis("on")

        points = plt.ginput(2, timeout=30)
        plt.close()

        if len(points) < 2:
            print("Region selection was not completed.")
            return None
        return points

    @staticmethod
    def _validate_coordinates(points):
        """Ensures that the selected coordinates are properly ordered."""
        (x1, y1), (x2, y2) = points
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Ensure correct ordering
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        return (x1, y1), (x2, y2)

    def get_background_pixel_value(self):
        """Public method to retrieve the stored mean pixel value."""
        if self._background_threshold_value is None:
            print("Background pixel value has not been computed yet.")
        return self._background_threshold_value

    def calculate_signal(self, image):
        """Calculate the percentage of pixels below the background threshold."""
        nr_of_pixels = image.size
        pixels_below_threshold = np.sum(image.ravel() < self._background_threshold_value)
        intensity_percentage = pixels_below_threshold / nr_of_pixels * 100
        return intensity_percentage

    def _process_image(self, input_image):
        if self._background_threshold_value is None:
            print("Selecting the background threshold")
            self._select_background(input_image)

        circular_mask = functions.create_circular_mask(self.config.patch_size, self.config.patch_size)
        filtered_image = ndimage.gaussian_filter(input_image, sigma=self.config.sigma)
        top_left_y, top_left_x, side_length = 0, 0, min(filtered_image.shape)

        n_w = int(1 + (side_length - self.config.patch_size) / self.config.patch_step)
        n_h = int(1 + (side_length - self.config.patch_size) / self.config.patch_step)

        count = 0
        for h in range(n_h):
            for w in range(n_w):
                vert_start = h * self.config.patch_step + top_left_y
                vert_end = vert_start + self.config.patch_size
                horiz_start = w * self.config.patch_step + top_left_x
                horiz_end = horiz_start + self.config.patch_size

                # Extract the sub-image
                sub_image = filtered_image[vert_start:vert_end, horiz_start:horiz_end]
                intensity_percentage = self.calculate_signal(sub_image)
                print(f"Patch nr {count} intensity pecentage {intensity_percentage}%")
                count += 1

    def process(self, image):
        """Main processing function (e.g., apply filtering)."""
        print(f"Processing image with sigma={self.config.sigma}")
        return self._process_image(image[0])
