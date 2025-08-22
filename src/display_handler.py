import matplotlib.pyplot as plt
import numpy as np
from src.radon_transform_algorithm.utils import functions, process_functions


class DisplayHandler:
    def __init__(self, display_mode="image"):
        self.display_mode = display_mode

    def display(self, results_dict: dict):
        """
        Displays results based on the chosen display_mode.

        :param results_dict: Dictionary containing result data
        :type results_dict: dict
        """
        mode_map = {
            "image": self._display_image,
            "polar_histogram": self._display_polar_histogram,
            "polar_histogram_over_days": self._display_polar_histogram_over_days,
            "widths": self._display_widths_distribution,
            "widths_over_days": self._display_widths_distribution_over_days,
        }

        display_function = mode_map.get(self.display_mode)
        if display_function:
            display_function(results_dict)
        else:
            print(f"Display mode '{self.display_mode}' is not supported yet.")

    def _display_image(self, results_dict: dict):
        """Helper function to display an image with detected lines and widths."""
        image_path = results_dict.get("input_data_path")
        image_array = process_functions.import_data(str(image_path))
        if image_array is None:
            print(f"Image not found at {image_path}.")

        counts, _ = np.histogram(image_array[0], bins=10)
        contrast_value = (np.argmax(counts) + 1) * 50

        plt.imshow(image_array[0], cmap='gray', vmax=contrast_value)  # Assuming grayscale images
        plt.axis("off")  # Hide axes for better visualization

        results = results_dict.get("results", [])
        if not results:
            print("No detection results found in results_dict.")
        else:
            for idx, point in enumerate(results):
                center = (point["x_center"], point["y_center"])
                angle_radians = point["angle_radians"]
                width = point["width"]

                # Compute endpoints for the detected line (blue)
                start, end = functions.compute_endpoints(center, 10, angle_radians)
                plt.plot([start[1], end[1]], [start[0], end[0]], linewidth=2, color='b')
                # plt.plot([point["y_start"], point["y_end"]], [point["x_start"], point["x_end"]], linewidth=2, color='b')

                # Mark the detected center with a small circle
                plt.plot(center[1], center[0], marker='o', markersize=2, color='cyan')

                # Compute endpoints for the width indicator (red, perpendicular to main line)
                wstart, wend = functions.compute_endpoints(center, int(width), angle_radians + np.pi / 2)
                plt.plot([wstart[1], wend[1]], [wstart[0], wend[0]], linewidth=1, color='r')

        plt.tight_layout()
        plt.show()

    def _display_polar_histogram(self, results_dict: dict):
        """Displays a polar histogram of angle distributions."""

        results = results_dict.get("results", [])
        if not results:
            print("No angle data found in results_dict.")
            return

        # Extract angle radians
        angles_deg = [entry["angle_degree"] for entry in results]

        # Create bins for histogram
        num_bins = 10  # Adjust based on desired bin size
        bins = np.linspace(0, 180, num_bins)

        # Create histogram
        counts, _ = np.histogram(angles_deg, bins=bins)
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

        plt.title(f"Polar Representation of Angle Distribution – {results_dict['input_data_name']}")
        plt.tight_layout()
        plt.show()

    def _display_polar_histogram_over_days(self, results_dict: dict):
        num_bins = 10  # Adjust based on desired bin size
        bins = np.linspace(0, 180, num_bins)

        hist_data = {}  # Store histogram data to avoid recomputation
        max_count = 0

        for day, details in results_dict.items():
            angles_deg = [entry['angle_degree'] for entry in details['results']]
            name = details['data_name'].replace("results_", "").split("-")[0]
            counts, _ = np.histogram(angles_deg, bins=bins)

            hist_data[day] = counts  # Store histogram counts
            max_count = max(max_count, max(counts))  # Track the highest bin count

        fig, axes = plt.subplots(1, len(results_dict), figsize=(4 * len(results_dict), 4),
                                 subplot_kw={'projection': 'polar'})
        fig.suptitle(name)

        if len(results_dict) == 1:
            axes = [axes]  # Ensure axes is iterable for a single day case

        bin_centers = np.radians(bins[:-1] + np.diff(bins) / 2)  # Compute once for all plots
        rad_bins = np.radians(bins)

        for ax, (day, counts) in zip(axes, hist_data.items()):
            # Plot histogram in polar coordinates
            ax.bar(bin_centers, counts,
                   width=np.diff(rad_bins),
                   bottom=0,
                   alpha=0.6,
                   color="#3FA592",
                   edgecolor="gray",
                   linewidth=1)

            # Format plot
            ax.set_theta_zero_location("E")  # 0 degrees at the right
            ax.set_theta_direction(1)  # Counterclockwise
            ax.set_thetamin(0)
            ax.set_thetamax(180)
            ax.set_rlim(0, max_count)  # Apply uniform r-limit
            ax.set_yticks(np.arange(0, max_count + 3, 3))
            ax.tick_params(axis="y", size=8)

            ax.set_title(day, fontsize=12)

        plt.tight_layout()
        # plt.show()

    def _display_widths_distribution(self, results_dict: dict):
        # Extract angle radians
        widths = [entry["width"] for entry in results_dict["results"]]

        plt.hist(widths, bins='sturges', edgecolor='black')  # 'auto' chooses the best bin size
        plt.xlabel("Width")
        plt.ylabel("Frequency")
        plt.title(f"Distribution of Structure Widths – {results_dict['input_data_name']}")
        plt.tight_layout()
        # plt.show()

    def _display_widths_distribution_over_days(self, results_dict: dict):
        # Create the widhts dict
        widths_dict = {}
        for day, details in results_dict.items():
            name = details['data_name'].replace("results_", "").split("-")[0]
            widths = [entry['width'] for entry in details['results']]
            widths_dict[day] = widths

        vals = (widths_dict.values())
        keys = (widths_dict.keys())

        plt.boxplot(vals, tick_labels=keys)
        # for i, group in enumerate(vals, start=1):  # Groups start at 1 in boxplot
        # x_values = np.random.normal(i, 0.05, size=len(group))  # Add jitter for scatter points
        # plt.scatter(x_values, group, alpha=0.7)

        plt.ylabel("Width")
        plt.title(name)
        plt.tight_layout()
        # plt.show()
