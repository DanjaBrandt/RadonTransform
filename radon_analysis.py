import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage.transform import radon
from skimage.draw import line
from scipy.signal import find_peaks
from scipy.spatial import KDTree

from utils import functions

import pprint #remove


class RadonStructureDetection:
    def __init__(self, config):
        self.config = config
        self._background_threshold_value = 30#None

    def _to_dict(self):
        """Returns a dictionary with all config parameters plus the threshold value."""
        return {
            **vars(self.config),  # Unpack all key-value pairs from self.config
            "background_threshold_value": self._background_threshold_value  # Add the threshold value
        }

    def _apply_gaussian_filter(self, image, sigma):
        """Apply Gaussian blur to the given image using sigma from the config."""
        return ndimage.gaussian_filter(image, sigma=sigma)

    def _set_background_threshold(self, input_image):
        """Ensures the background threshold is selected before proceeding."""
        if self._background_threshold_value is None:
            print("Selecting the background threshold")
            self._select_background(input_image)

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

    def _calculate_signal(self, image):
        """Calculate the percentage of pixels below the background threshold."""
        nr_of_pixels = image.size
        pixels_below_threshold = np.sum(image.ravel() <= self._background_threshold_value)
        intensity_percentage = pixels_below_threshold / nr_of_pixels * 100
        return intensity_percentage

    def _calculate_width(self, max_projection_plot):
        """Computes the Full Width at Half Maximum (FWHM) of the strongest peak in the given plot."""

        # Smooth the data
        data_1d = self._apply_gaussian_filter(max_projection_plot, sigma=self.config.sigma_1d).flatten()

        # Find extrema and boundaries
        max_peak_idx, main_min_idx, left_bound, right_bound = self._find_peak_and_boundaries(data_1d)

        # Extract the relevant peak region
        peak_values = self._extract_peak_region(data_1d, max_peak_idx, main_min_idx, left_bound, right_bound)

        # Calculate and return the FWHM
        return self._compute_fwhm(peak_values)

    @staticmethod
    def _find_peak_and_boundaries(data_1d):
        """Finds the highest peak and its surrounding boundaries."""
        local_max, local_min = functions.find_local_extrema(data_1d)
        local_min = np.concatenate(([0], local_min, [len(data_1d) - 1]))

        if local_max.size == 0:
            local_max = [np.argmax(data_1d)]
            #plt.plot(data_1d)
            #plt.show()

        max_peak_idx = max(local_max, key=lambda i: data_1d[i])
        left_bound, right_bound = functions.find_bounds(max_peak_idx, local_min)
        main_min_idx = max((left_bound, right_bound), key=lambda i: data_1d[i])

        return max_peak_idx, main_min_idx, left_bound, right_bound

    @staticmethod
    def _extract_peak_region(data_1d, max_peak_idx, main_min_idx, left_bound, right_bound):
        """Extracts the peak region based on the threshold mask."""
        peak_mask = data_1d > data_1d[main_min_idx]

        if main_min_idx > max_peak_idx:
            sub_data = data_1d[left_bound:main_min_idx]
            masked_sub_data = sub_data[peak_mask[left_bound:main_min_idx]]
            return np.insert(masked_sub_data, 0, data_1d[main_min_idx])
        else:
            sub_data = data_1d[main_min_idx:right_bound]
            masked_sub_data = sub_data[peak_mask[main_min_idx:right_bound]]
            return np.insert(masked_sub_data, len(masked_sub_data), data_1d[main_min_idx])

    @staticmethod
    def _compute_fwhm(peak_values):
        """Computes the Full Width at Half Maximum (FWHM)."""
        return functions.calculate_fwhm(np.arange(len(peak_values)), peak_values)[0]

    def _analyze_sinogram(self, input_sinogram):
        """
        Analyzes a sinogram to detect peaks and extract relevant information.

        :param input_sinogram: 2D array representing the sinogram.
        :return: List of dictionaries containing peak details or None if no peaks are found.
        """
        # Compute max intensity projection and key values
        max_projection = input_sinogram.max(axis=0)
        max_sinogram_value = max_projection.max()

        # Define thresholds for peak detection
        peak_threshold = max_sinogram_value * self.config.peak_threshold
        peak_prominence = max_sinogram_value * self.config.peak_prominence
        border_limit = 10

        # Detect peaks in the sinogram projection
        peaks, _ = find_peaks(max_projection, prominence=peak_prominence, height=peak_threshold)

        # Handle boundary peaks efficiently
        left_border_peak = np.argmax(max_projection[:border_limit]) if max_projection[
                                                                       :border_limit].max() == max_sinogram_value else None
        right_border_peak = len(max_projection) - border_limit + np.argmax(
            max_projection[-border_limit:]) if max_projection[-border_limit:].max() == max_sinogram_value else None

        # Append detected border peaks (if any)
        if left_border_peak is not None:
            peaks = np.insert(peaks, 0, left_border_peak)
        if right_border_peak is not None:
            peaks = np.insert(peaks, len(peaks), right_border_peak)

        # If no peaks are found, return None
        if peaks.size == 0:
            return None

        # Extract sinogram features for each detected peak
        return [
            {
                "sinogram_x": np.argmax(input_sinogram[:, peak if peak <= 179 else peak - 180]),
                "sinogram_deg_angle": peak if peak <= 179 else peak - 180,
                "image_rad_angle": np.radians(peak + 90),
                "width": self._calculate_width(input_sinogram[:, peak if peak <= 179 else peak - 180])
            }
            for peak in peaks
        ]

    def _analyze_profile(self, underlying_image, p1, p2):
        """
        Analyzes a profile along a line in the image and determines if it passes the background threshold.

        Parameters:
        - underlying_image (numpy.ndarray): The image in which the profile is analyzed.
        - p1 (tuple): Starting point (x, y) of the line.
        - p2 (tuple): Ending point (x, y) of the line.

        Returns:
        - tuple: (p1, p2) if the profile meets the threshold, otherwise None.
        """

        x1, y1 = map(int, p1)  # Convert p1 (x, y) → (y1, x1)
        x2, y2 = map(int, p2)  # Convert p2 (x, y) → (y2, x2)

        # Generate pixel indices along the line
        rr, cc = line(y1, x1, y2, x2)

        # Track the last valid point before hitting the threshold
        last_valid_y, last_valid_x = y1, x1

        # Iterate over the line points
        for y, x in zip(rr, cc):
            # Ensure coordinates are within image bounds before accessing pixel values
            if 0 <= x < underlying_image.shape[1] and 0 <= y < underlying_image.shape[0]:

                if underlying_image[y, x] <= self._background_threshold_value:
                    # return p1, (last_valid_y, last_valid_x)
                    return None
            last_valid_y, last_valid_x = y, x
        return p1, p2

    def _process_sinogram_peaks(self, sub_image, sinogram_peaks, x, y, final_results):
        """Analyzes peaks found in the sinogram and extracts feature points."""
        for peak in sinogram_peaks:
            subimage_center = functions.convert_coord(sub_image.shape, peak["sinogram_x"], peak["sinogram_deg_angle"])
            p_start, p_end = functions.find_circle_line_intersections(
                (sub_image.shape[0] // 2, sub_image.shape[1] // 2),
                sub_image.shape[0] // 2 - 1,
                subimage_center,
                peak["image_rad_angle"]
            )
            if p_start and p_end:
                passed_line_coords = self._analyze_profile(sub_image, p_start, p_end)
                if passed_line_coords:
                    self._store_results(passed_line_coords, subimage_center, x, y, peak, final_results)

    @staticmethod
    def _store_results(passed_line_coords, subimage_center, x, y, peak, final_results):
        """Stores computed patch results."""
        final_center = (subimage_center[0] + y, subimage_center[1] + x)
        final_p_start = (passed_line_coords[0][1] + y, passed_line_coords[0][0] + x)
        final_p_end = (passed_line_coords[1][1] + y, passed_line_coords[1][0] + x)
        angle_degree = np.degrees(peak["image_rad_angle"])
        if angle_degree >= 180:
            angle_degree = abs(np.degrees(np.pi - peak["image_rad_angle"]))

        final_results.append({
            "center": final_center,
            "p_start": final_p_start,
            "p_end": final_p_end,
            "angle_radians": float(peak["image_rad_angle"]),
            "angle_degree": float(angle_degree),
            "width": float(peak["width"])
        })

    def _group_points(self, total_points):
        """Groups close and similarly oriented points into single averaged points."""
        if not total_points:
            return []

        def combine_group(group):
            """Computes mean position and orientation for grouped points."""
            group_array = np.array([[p['center'][0], p['center'][1],
                                     p['p_start'][0], p['p_start'][1],
                                     p['p_end'][0], p['p_end'][1],
                                     p['angle_radians'], p['angle_degree'],
                                     p['width']]
                                    for p in group])
            result = {
                "center": tuple(group_array[:, :2].mean(axis=0)),
                "p_start": tuple(group_array[:, 2:4].mean(axis=0)),
                "p_end": tuple(group_array[:, 4:6].mean(axis=0)),
                "angle_radians": group_array[:, 6].mean(),
                "angle_degree": group_array[:, 7].mean(),
                "width": group_array[:, 8].max()  # Take the max width
            }

            return result

        kdtree, point_coords = KDTree([(p['center'][1], p['center'][0]) for p in total_points]), []
        grouped, used = [], set()

        for i, point in enumerate(total_points):
            if i in used:
                continue
            neighbors = [total_points[j] for j in kdtree.query_ball_point((point['center'][1], point['center'][0]),
                                                                          self.config.distance_threshold)
                         if j != i and j not in used and min(
                    abs(point['angle_radians'] - total_points[j]['angle_radians']),
                    np.pi - abs(point['angle_radians'] - total_points[j]['angle_radians'])) < np.radians(
                    self.config.angle_threshold)]
            [used.add(total_points.index(n)) for n in neighbors]
            grouped.append(combine_group([point] + neighbors))

        return grouped

    def _process_single_patch(self, sub_image, x, y, circular_mask, final_results):
        """Processes a single image patch."""
        intensity_percentage = self._calculate_signal(sub_image)
        #print(f"Patch ({x}, {y}): Intensity = {intensity_percentage:.2f}%")  # REMOVE LATER

        if intensity_percentage < self.config.background_pixel_cutoff:
            self.display_patches.append(sub_image)
            mask_image = sub_image * circular_mask
            sinogram = radon(mask_image)
            sinogram_peaks = self._analyze_sinogram(sinogram)

            if sinogram_peaks:
                self._process_sinogram_peaks(sub_image, sinogram_peaks, x, y, final_results)
            else:
                print("No peaks detected.")
        else:
            print(f"Subimage ({x}, {y}) mostly empty.")

    def _process_patches(self, filtered_image):
        """Extracts patches, applies a circular mask, and stores the patch if it meets the threshold."""

        # Ensure valid patch configuration
        if self.config.patch_step <= 0 or self.config.patch_size > min(filtered_image.shape):
            print("Invalid patch size or step configuration.")
            return

        self.display_patches = []  # Store patches that meet the threshold
        circular_mask = functions.create_circular_mask(self.config.patch_size, self.config.patch_size)

        patch_indices_y = np.arange(0, filtered_image.shape[0] - self.config.patch_size + 1, self.config.patch_step)
        patch_indices_x = np.arange(0, filtered_image.shape[1] - self.config.patch_size + 1, self.config.patch_step)

        raw_results = []
        for y in patch_indices_y:
            for x in patch_indices_x:
                sub_image = filtered_image[y:y + self.config.patch_size, x:x + self.config.patch_size]
                self._process_single_patch(sub_image, x, y, circular_mask, raw_results)

        # Apply point grouping after all patches are processed
        grouped_results = self._group_points(raw_results)

        return grouped_results

    def _process_image(self, input_image):
        """Processes an image by applying a filter and analyzing patches."""

        # Ensure background threshold is set
        self._set_background_threshold(input_image)

        # Apply Gaussian filter
        filtered_image = self._apply_gaussian_filter(input_image, sigma=self.config.sigma)

        # Process patches
        results = self._process_patches(filtered_image)

        used_config = self._to_dict()
        # Reset the threshold after each image
        #self._background_threshold_value = None

        plot_detected_features(filtered_image, results)
        return used_config, results

    def process(self, image):
        """Main processing function (e.g., apply filtering)."""
        return self._process_image(image[0])


# REMOVE WHEN DONE
def show_patches(self):
    """Displays the extracted patches in a grid layout."""

    if not self.display_patches:
        print("No patches available for display.")
        return

    num_patches = len(self.display_patches)
    grid_size = int(np.ceil(np.sqrt(num_patches)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(5, 5))

    for i, ax in enumerate(axes.flat):
        if i < num_patches:  # Ensure we don't access out-of-bounds
            ax.imshow(self.display_patches[i], cmap="gray")
            ax.axis("off")  # Hide axes
        else:
            ax.axis("off")  # Hide extra empty subplots

    plt.tight_layout()
    plt.show()


def plot_detected_features(image, final_results):
    """
    Plots the detected features on the image.

    Parameters:
        image (np.array): The image to display.
        final_results (list): List of detected feature dictionaries containing center, angle, and width.
    """
    plt.imshow(image, cmap='gray')  # Display the image
    plt.axis('off')

    for idx, point in enumerate(final_results):
        center = point["center"]
        angle_radians = point["angle_radians"]
        width = point["width"]

        # Compute endpoints for the detected line (blue)
        start, end = functions.compute_endpoints(center, 10, angle_radians)
        plt.plot([start[1], end[1]], [start[0], end[0]], linewidth=2, color='b')

        # Mark the detected center with a small circle
        plt.plot(center[1], center[0], marker='o', markersize=2, color='cyan')

        # Compute endpoints for the width indicator (red, perpendicular to main line)
        wstart, wend = functions.compute_endpoints(center, int(width), angle_radians + np.pi / 2)
        plt.plot([wstart[1], wend[1]], [wstart[0], wend[0]], linewidth=1, color='r')

        # Add a label near the center for reference
        # plt.text(center[1] + random.randint(1, 3), center[0] + random.randint(1, 3),
        # str(idx), fontsize=9, color='white')

    plt.show()
