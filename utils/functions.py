import numpy as np
import os
from scipy.interpolate import interp1d
import json
import csv
import copy


def calculate_fwhm(x: list[float], y: list[float]) -> tuple[float, float, float, float]:
    """
    Calculate the Full Width at Half Maximum (FWHM) of a peak
    :param x: The x-values of the data points
    :param y: The y-values of the data points
    :return: A tuple containing:
            - fwhm (float): The full width at half maximum.
            - peak_value (float): The peak value of the y-data.
            - left_cross (float): The x-value at the left half maximum.
            - right_cross (float): The x-value at the right half maximum.
    """
    # Find the maximum value and its position
    peak_index = np.argmax(y)
    peak_value = y[peak_index]

    # Calculate half maximum
    half_max = peak_value / 2.0

    # Find the indices where the data crosses the half maximum
    indices_above_half_max = np.where(y >= half_max)[0]

    # Identify the left and right crossing points
    left_index = indices_above_half_max[0]
    right_index = indices_above_half_max[-1]

    # Handle edge cases: left_index == 0 or right_index == len(data) - 1
    if left_index == 0:
        left_cross = x[left_index]
    else:
        interp_func_left = interp1d(y[left_index - 1:left_index + 1], x[left_index - 1:left_index + 1])
        left_cross = interp_func_left(half_max)

    if right_index == len(y) - 1:
        right_cross = x[right_index]
    else:
        interp_func_right = interp1d(y[right_index:right_index + 2], x[right_index:right_index + 2])
        right_cross = interp_func_right(half_max)

    # Calculate FWHM
    fwhm = right_cross - left_cross

    return fwhm, peak_value, left_cross, right_cross


def compute_endpoints(center: tuple[float, float], length: float, angle_rad: float) -> tuple[tuple[float, float],
                                                                                             tuple[float, float]]:
    """
    Compute the endpoints of a line segment given its center, length, and angle
    :param center: The (x, y) coordinates of the center of the line segment
    :param length: The length of the line segment
    :param angle_rad: The angle of the line segment in radians, measured from the positive x-axis
    :return: A tuple containing:
            - start_point (Tuple[float, float]): The (x, y) coordinates of the start point.
            - end_point (Tuple[float, float]): The (x, y) coordinates of the end point.
    """
    dx = length * np.cos(angle_rad) / 2
    dy = length * np.sin(angle_rad) / 2

    start_point = (center[0] + dy, center[1] - dx)
    end_point = (center[0] - dy, center[1] + dx)
    return start_point, end_point


def convert_coord(image_shape, x_prime, theta):
    """
    Converts coordinates from the x'y' reference system used in sinogram computation to the xy reference system of
    the input image
    :param image: image used for the sinogram computation
    :param x_prime: max value found on the x' axis
    :param theta: angle of the max value in the sinogram image
    :return: coordinates of the max value on the input image
    """
    img_h, img_w = image_shape
    c_x, c_y = img_h // 2, img_w // 2
    r = img_w // 2
    theta_rad = np.radians(theta)

    # Calculate the position of O' on the circumference
    t = np.array([c_x - r * np.cos(theta_rad), c_y + r * np.sin(theta_rad)])

    # Rotation matrix for the angle theta (same as alpha)
    r = np.array([
        [np.cos(theta_rad), -np.sin(theta_rad)],
        [-np.sin(theta_rad), np.cos(theta_rad)]
    ])

    # Coordinates in the x'y' system
    p_prime = np.array([x_prime, 0])

    # Rotate the coordinates
    p_rotated = r @ p_prime

    # Translate the coordinates
    p = p_rotated + t

    # Extract x and y coordinates
    x = p[0]
    y = p[1]

    return y, x


def create_circular_mask(height: int, width: int, center: tuple = None, radius: int = None) -> np.ndarray:
    """
     Creates a circular mask
    :param height: Height of the mask
    :param width: Width of the mask
    :param center: Center of the circle. Defaults to the middle of the image
    :param radius: Radius of the circle. Defaults to the smallest distance between
                               the center and image borders
    :return: Circular mask
    """

    if center is None:
        center = (width // 2, height // 2)

    if radius is None:
        radius = min(center[0], center[1], width - center[0], height - center[1])

    y, x = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

    mask = dist_from_center <= radius

    return mask


def get_unique_name(base_name: str, extension: str = None) -> str:
    """
    Generates a unique name for a folder or file by appending numbers (_01, _02, ...).

    :param base_name: The base name (e.g., "radon_output" or "logs/radon_log").
    :param extension: Optional file extension (e.g., ".log"). If None, assumes it's a folder.
    :return: A unique folder or file name.
    """
    if extension:
        name = f"{base_name}{extension}"
    else:
        name = base_name  # Folder case

    if not os.path.exists(name):
        return name  # If it doesn't exist, return as is

    counter = 1
    while True:
        new_name = f"{base_name}_{counter:02d}{extension if extension else ''}"
        if not os.path.exists(new_name):
            return new_name  # Return the first available unique name
        counter += 1


def find_bounds(num: float, lst: list[float]) -> tuple[[float], [float]]:
    """
    Find the bounds in a sorted list within which a given number falls
    :param num: Find the bounds in a sorted list within which a given number falls.
    :param lst: The list of numbers to search, which does not need to be sorted
    :return: A tuple containing:
            - lower_bound (Optional[float]): The largest number in the list that is less than or equal to `num`.
            - upper_bound (Optional[float]): The smallest number in the list that is greater than or equal to `num`.
    """
    # Sort the list
    sorted_lst = sorted(lst)

    # Initialize bounds
    lower_bound = None
    upper_bound = None

    # Find the bounds
    for i in range(len(sorted_lst) - 1):
        if sorted_lst[i] <= num <= sorted_lst[i + 1]:
            lower_bound = sorted_lst[i]
            upper_bound = sorted_lst[i + 1]
            break

    return [lower_bound, upper_bound]


def find_circle_line_intersections(circle_center, radius, point, angle):
    """
    Finds the intersection points of a line with a given circle.

    - circle_center (tuple): (x_c, y_c), coordinates of the circle's center.
    - radius (float): Radius of the circle.
    - point (tuple): (y_0, x_0), a point on the line.
    - angle (float): Angle of the line in radians.

    Returns:
    - tuple: Two intersection points (extreme_point1, extreme_point2),
             or (None, None) if no intersections exist.
    """

    # Unpack center and point coordinates
    x_c, y_c = circle_center
    y_0, x_0 = point

    # Compute direction vector for the line based on the given angle
    dx = np.cos(angle)
    dy = -np.sin(angle)  # Negative sign to align with image coordinate systems

    # Quadratic equation coefficients (for solving intersection points)
    A = dx ** 2 + dy ** 2  # Always 1 due to unit direction vector
    B = 2 * ((x_0 - x_c) * dx + (y_0 - y_c) * dy)
    C = (x_0 - x_c) ** 2 + (y_0 - y_c) ** 2 - radius ** 2

    # Compute discriminant to check for real solutions
    discriminant = B ** 2 - 4 * A * C
    if discriminant < 0:
        return None, None  # No real intersection points

    # Compute the intersection parameters (t1 and t2)
    sqrt_discriminant = np.sqrt(discriminant)
    t1 = (-B + sqrt_discriminant) / (2 * A)
    t2 = (-B - sqrt_discriminant) / (2 * A)

    # Compute the intersection points using t1 and t2
    extreme_point1 = (x_0 + t1 * dx, y_0 + t1 * dy)
    extreme_point2 = (x_0 + t2 * dx, y_0 + t2 * dy)

    return extreme_point1, extreme_point2


def find_local_extrema(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Find local maxima and minima indices in a 1D array of data
    :param data: 1D array containing the data points
    :return: A tuple containing:
            - local_maxs (np.ndarray): Indices of local maxima in the data array
            - local_mins (np.ndarray): Indices of local minima in the data array
    """
    second_derivative = np.diff(data)
    sign_second_derivative = np.sign(second_derivative)
    diff_sign_second_derivative = np.diff(sign_second_derivative)

    local_maxs = np.where(diff_sign_second_derivative < 0)[0] + 1
    local_mins = np.where(diff_sign_second_derivative > 0)[0] + 1

    return local_maxs, local_mins


def reformat_data(data):
    formatted_data = []

    for entry in data:
        formatted_entry = {
            "x_center": entry["center"][0],
            "y_center": entry["center"][1],
            "x_start": entry["p_start"][0],
            "y_start": entry["p_start"][1],
            "x_end": entry["p_end"][0],
            "y_end": entry["p_end"][1],
            "angle_radians": entry["angle_radians"],
            "angle_degree": entry["angle_degree"],
            "width": entry["width"],
        }
        formatted_data.append(formatted_entry)

    return formatted_data


def write_config(data_name, config_data, output_path):
    """
    Saves config and threshold to JSON.

    Parameters:
        data_name (str): The name of the dataset.
        config_file (dict): The config dict.
        output_path (str): The directory where files will be saved
    """

    # Define file paths
    json_filename = os.path.join(output_path, f"{data_name}.json")

    def convert_np_types(obj):
        """Converts NumPy types to standard Python types for JSON serialization."""
        if isinstance(obj, np.generic):  # Handles np.float64, np.int64, etc.
            return obj.item()
        return obj

    # Save JSON file
    with open(json_filename, "w") as json_file:
        json.dump(config_data, json_file, indent=4, default=convert_np_types)

    print(f"Config saved to {json_filename}")


def write_results(data_name, results, output_path):
    """
    Saves results to JSON and CSV files.

    Parameters:
        data_name (str): The name of the dataset.
        results (list of dict): The result data.
        output_path (str): The directory where files will be saved.
    """

    formatted_results = reformat_data(results)

    # Convert NumPy floats to standard floats
    cleaned_results = [{key: float(value) for key, value in result.items()} for result in formatted_results]

    # Define file paths
    json_filename = os.path.join(output_path, f"{data_name}.json")
    csv_filename = os.path.join(output_path, f"{data_name}.csv")

    # Save JSON file
    json_data = {"data_name": data_name, "results": cleaned_results}
    with open(json_filename, "w") as json_file:
        json.dump(json_data, json_file, indent=4)

    print(f"Saved JSON to {json_filename}")

    # Save CSV file
    csv_headers = cleaned_results[0].keys() if cleaned_results else []
    with open(csv_filename, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_headers)
        writer.writeheader()
        writer.writerows(cleaned_results)

    print(f"Saved CSV to {csv_filename}")
