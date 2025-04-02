import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from utils import functions

from utils import process_functions


class DisplayHandler:
    def __init__(self, display_mode="image"):
        self.display_mode = display_mode

    def display(self, results_dict: dict):
        """
        Displays an image using Matplotlib.

        :param image_array: Image data as a NumPy array
        :type image_array: np.ndarray
        :param results_dict: Dictionary containing result data
        :type results_dict: dict
        """
        if self.display_mode == "image":
            _display_image(results_dict)
        elif self.display_mode == "polar_histogram":
            _display_polar_histogram(results_dict)
        else:
            print(f"Display mode '{self.display_mode}' is not supported yet.")


def _display_image(results_dict: dict):
    """Helper function to display an image."""
    image_path = results_dict.get("input_data_path")
    image_array = process_functions.import_data(str(image_path))
    if image_array is None:
        print(f"Image not found at {image_path}.")

    plt.imshow(image_array[0], cmap='gray')  # Assuming grayscale images
    plt.axis("off")  # Hide axes for better visualization
    # plt.show()
    for idx, point in enumerate(results_dict["results"]):
        center = (point["x_center"], point["y_center"])
        angle_radians = point["angle_radians"]
        width = point["width"]

        # Compute endpoints for the detected line (blue)
        start, end = functions.compute_endpoints(center, 10, angle_radians)
        plt.plot([start[1], end[1]], [start[0], end[0]], linewidth=2, color='b')
        #plt.plot([point["y_start"], point["y_end"]], [point["x_start"], point["x_end"]], linewidth=2, color='b')

        # Mark the detected center with a small circle
        plt.plot(center[1], center[0], marker='o', markersize=2, color='cyan')

        # Compute endpoints for the width indicator (red, perpendicular to main line)
        wstart, wend = functions.compute_endpoints(center, int(width), angle_radians + np.pi / 2)
        plt.plot([wstart[1], wend[1]], [wstart[0], wend[0]], linewidth=1, color='r')
    plt.show()


def _display_polar_histogram(results_dict: dict):
    # Extract angle radians
    angles_radians = [entry["angle_degree"] for entry in results_dict["results"]]

    # Create bins for histogram
    num_bins = 10  # Adjust based on desired bin size
    bins = np.linspace(0, 180, num_bins)

    # Create histogram
    counts, _ = np.histogram(angles_radians, bins=bins)
    print('bins', bins)
    print('counts', counts)
    bin_centers = np.radians(bins[:-1] + np.diff(bins) / 2)
    rad_bins = np.radians(bins)

    # Polar plot
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    # Plot histogram in polar coordinates
    bars = ax.bar(bin_centers, counts,
                  width=np.diff(rad_bins),
                  bottom=0,
                  alpha=0.6,
                  color="#3FA592",
                  edgecolor="gray",
                  linewidth=1
                  )

    # Format plot
    ax.set_theta_zero_location("E")  # 0 degrees at the top
    ax.set_theta_direction(1)  # Counterclockwise
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    #ax.set_rlim(0, 40)
    plt.title(f"Polar Representation of Angle Distribution – {results_dict['input_data_name']}")
    plt.show()
