import matplotlib.pyplot as plt
import os
from PIL import Image


class DisplayHandler:
    def __init__(self, display_mode="image"):
        self.display_mode = display_mode

    def display(self, image_array: str):
        """
        Displays an image using Matplotlib.

        :param image_path: Path to the image file
        :type image_path: str
        """
        if self.display_mode == "image":
            self._display_image(image_array)
        else:
            print(f"Display mode '{self.display_mode}' is not supported yet.")

    def _display_image(self, image_array: str):
        """Helper function to display an image."""
        #if not os.path.exists(image_array):
            #print(f"File not found: {image_array}")
            #return

        try:
            #image = Image.open(image_path)
            plt.imshow(image_array[0], cmap='gray')  # Assuming grayscale images
            plt.axis("off")  # Hide axes for better visualization
            plt.show()
        except Exception as e:
            print(f"Error displaying image: {e}")