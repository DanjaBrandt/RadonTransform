import os

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Callable

#from .logging_config import logger


def import_data(path: str, normalize: bool = True, max_value: int = 255, min_value: int = 0) -> np.ndarray:
    """
    Opens an image or a stack and returns the bits of the image,
    the array of the image and the shape of the array
    :param path: Path to the file to be read
    :type path: str
    :param normalize: if True outputs image normalized between max_value and min_value
    :type normalize: bool
    :param max_value: max value of grayscale values
    :type max_value: int
    :param min_value: min value of grayscale values
    :type min_value: int
    :return: the image stack (if just 1 img shape:[1,:,:,:])
    :rtype: np.ndarray
    """

    try:
        # Read PIL image
        with Image.open(path) as img:
            # Read image depth
            mode = img.mode
            '''
            if mode.startswith('L'):
                logger.info(f"{mode}: 8-bit grayscale imported data")
            elif mode.startswith('I;16'):
                logger.info(f"{mode}: 16-bit grayscale imported data")
            elif mode.startswith('RGB'):
                logger.info(f"{mode}: 24-bit color imported data")
                img = img.convert('L')
                logger.info("Imported RGB image converted to 8-bit grayscale image")
            else:
                logger.info(f"{mode}: Unknown image depth for PIL type")'''

            # Create image array
            h, w = img.size
            imarray = np.array(img)
            tiffarray = np.zeros((img.n_frames, h, w))
            for i in range(img.n_frames):
                img.seek(i)
                # Normalize the image if normalized=True
                if normalize:
                    #logger.info(f'Image is normalized between {min_value} and {max_value}')
                    norm = ((imarray - np.min(imarray)) * (
                            (max_value - min_value) / (np.max(imarray) - np.min(imarray))) + min_value)
                    tiffarray[i, :, :] = norm
                else:
                    #logger.info('Attention! Image is not normalized')
                    tiffarray[i, :, :] = imarray

            #logger.info(f'Imported data of shape: {tiffarray.shape}. Data-type: {tiffarray[0][0][0].dtype}')

    except FileNotFoundError:
        #logger.error("Data file not found")
        print(("Data file not found"))

    return tiffarray


def process_images_in_folder(input_folder: str, process_function: Callable, process_key: str, normalize: bool = True):
    """
    Processes all images in a given folder and saves the results.

    The function:
    - Recursively scans the input folder.
    - Opens each image using `import_data()`.
    - Applies the specified `process_function()`.
    - Saves the processed image to `output_folder` (if specified), maintaining folder structure.

    :param input_folder: The root folder containing input images.
    :type input_folder: str
    :param process_function: A function that processes a NumPy array and returns an output.
    :type process_function: Callable
    :param process_key: The name of the process function (Radon, Fourier...)
    :type normalize: object
    """

    base_output_folder = f"{process_key}_output"
    output_folder = get_unique_output_folder(base_output_folder)
    for root, _, files in os.walk(input_folder):

        for file in files:
            file_path = os.path.join(root, file)

            # Try importing the image
            image_array = import_data(file_path, normalize=normalize)
            if image_array is None:
                continue  # Skip if not a valid image

            processed_image = process_function(image_array)

            # Save the processed image if process_key is not display
            if process_key != "display_input":
                # Preserve subfolder structure
                relative_path = os.path.relpath(root, input_folder)
                output_dir = os.path.join(output_folder, relative_path)
                #os.makedirs(output_dir, exist_ok=True)
                save_path = os.path.join(output_folder, relative_path)
                #os.makedirs(save_path, exist_ok=True)

                output_file_path = os.path.join(save_path, file)


def display_outputs_in_folder(input_folder: str, process_function: Callable, display_mode: str):
    """
    Displays all images in a given folder.

    Parameters:
    - input_folder (str): Folder containing images to display.

    Returns:
    - None
    """
    for root, _, files in os.walk(input_folder):
        for file in files:
            file_path = os.path.join(root, file)

            # Check if it's an image
            #if imghdr.what(file_path) is None:
                #continue  # Skip non-image files

            #BASED ON DISPLAY MODE OPEN IMAGE; OR CSV OR OTHER RESULT
            # Load and display the image
            image_array = import_data(file_path)
            if image_array is None:
                continue

            plt.imshow(image_array[0], cmap="gray")  # Display first frame if multi-frame
            plt.title(f"Displaying: {file}")
            plt.show()


def get_unique_output_folder(base_name: str) -> str:
    """
    Checks if the output folder already exists and finds a unique name by appending numbers (_01, _02, ...).

    :param base_name: The base name of the folder (e.g., "radon_output")
    :return: A unique folder name (e.g., "radon_output_01" or "radon_output_02")
    """
    if not os.path.exists(base_name):
        return base_name  # If it doesn't exist, use the base name

    counter = 1
    while True:
        new_folder = f"{base_name}_{counter:02d}"
        if not os.path.exists(new_folder):
            return new_folder  # Return the first available unique name
        counter += 1