import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import copy
import seaborn as sns
import pandas as pd
import pprint
from typing import Dict, Any

from src.radon_transform_algorithm.utils import functions, process_functions


class DisplayGrowingPlate:
    def __init__(self, **kwargs):
        self.display_mode = kwargs.get("display_mode", "images")

        self.group = kwargs.get("group", None)
        self.mouse = kwargs.get("mouse", None)
        self.masked_region = kwargs.get("masked_region", None)

    def display(self, results_dict: dict, masks_dict: dict):
        """
        Displays results based on the chosen display_mode.

        :param results_dict: Dictionary containing result data
        :param masks_dict: Dictionary containing the paths to the masks
        :type results_dict: dict
        """
        mode_map = {
            "image": self._display_image,
            "detected_points": self._display_points_plot,
            "angle_alignment": self._display_angle_alignment,
        }

        display_function = mode_map.get(self.display_mode)
        if self.masked_region:
            results_dict = self._filter_results_by_mask(results_dict, masks_dict)
        else:
            print('No mask provided, all results will be used')

        if display_function:
            display_function(results_dict)
        else:
            print(f"Display mode '{self.display_mode}' is not supported yet.")

    def _display_image(self, results_dict: dict):
        """Helper function to display an image with detected lines and widths."""
        single_result = results_dict[self.group][self.mouse]
        image_path = single_result.get("input_data_path")
        image_array = process_functions.import_data(str(image_path))
        if image_array is None:
            print(f"Image not found at {image_path}.")

        plt.imshow(image_array[0], cmap='gray')
        # plt.imshow(self.debug_mask, cmap='gray', alpha=0.5)
        # plt.axis("off")  # Hide axes for better visualization

        results = single_result.get("results", [])
        if not results:
            print("No detection results found in results_dict.")
        else:
            for idx, point in enumerate(results):
                center = (point["x_center"], point["y_center"])
                angle_radians = point["angle_radians"]
                width = point["width"]

                # Compute endpoints for the detected line (blue)
                start, end = functions.compute_endpoints(center, 50, angle_radians)
                plt.plot([start[1], end[1]], [start[0], end[0]], linewidth=2, color='b')

                # Mark the detected center with a small circle
                plt.plot(center[1], center[0], marker='o', markersize=2, color='cyan')

                # Compute endpoints for the width indicator (red, perpendicular to main line)
                wstart, wend = functions.compute_endpoints(center, int(width), angle_radians + np.pi / 2)
                plt.plot([wstart[1], wend[1]], [wstart[0], wend[0]], linewidth=1, color='r')

        plt.tight_layout()

    def _display_points_plot(self, results_dict: Dict[str, Dict[str, Any]]) -> None:
        """
        Create and display a box + strip plot of detected points per group.

        Args:
            results_dict (dict): Nested dictionary with detection results.
        """
        # --- Count points ---
        points_dict = self._count_points(results_dict)
        # pprint.pprint(signal_plot)

        # --- Flatten into list of records for DataFrame ---
        records = [
            {"group": group, "subject": subject, "detected_points": count}
            for group, subjects in points_dict.items()
            for subject, count in subjects.items()
        ]
        df = pd.DataFrame(records)

        if df.empty:
            print("⚠️ No data available to plot.")
            return

        # --- Ensure group order ---
        expected_order = ["young", "middle", "old"]
        df["group"] = pd.Categorical(df["group"], categories=expected_order, ordered=True)

        # --- Set palette ---
        palette = sns.color_palette("tab10", n_colors=df["subject"].nunique())

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(6, 4))

        sns.boxplot(
            x="group", y="detected_points", data=df,
            ax=ax, color="lightgray"
        )
        sns.stripplot(
            x="group", y="detected_points", data=df,
            hue="subject", ax=ax, jitter=True, size=5,
            palette=palette, dodge=True, legend=False
        )

        ax.set_title("Detected Points by Group")
        ax.set_ylabel("Detected Points")

        plt.tight_layout()

        # if self.display_parameters["save"]:
        # save_path = self.display_parameters["image_saving_folder"]
        # saving_name = f"points_analysis_{self.radon.config.patch_size}_all"
        # plt.savefig(os.path.join(save_path, saving_name), bbox_inches='tight', pad_inches=0, dpi=300)

    def _display_angle_alignment(self, results_dict):
        """
        Display polar histograms of cortex orientation alignment for a given mouse.

        Parameters
        ----------
        results_dict : dict
            Dictionary containing angle results for groups and mice.
        """
        # --- Extract results for this mouse ---
        try:
            single_result = results_dict[self.group][self.mouse]['results']
        except KeyError:
            raise ValueError(f"No results found for group '{self.group}' and mouse '{self.mouse}'")

        # --- Load reference orientation values ---
        csv_path = "C:/Users/Danja Brandt/Desktop/FU/SideProjects/George/cortex_dir.csv"
        reference_values = self._import_orientation_values(csv_path)

        if reference_values.empty:
            raise ValueError("Reference values CSV is empty")

        # --- Get mouse name ---
        try:
            mouse_name = reference_values.iloc[0]["mouse"]
        except KeyError:
            raise ValueError("CSV must contain a 'mouse' column")

        # --- Extract all cortex direction columns ---
        cortex_columns = [col for col in reference_values.columns if "_cortex_dir" in col]
        if not cortex_columns:
            raise ValueError("No cortex direction columns found in reference values CSV")

        # --- Extract angle measurements ---
        angles = [float(item['angle_degree']) for item in single_result]
        angles_deg = np.array(angles)

        # --- Create subplots dynamically ---
        num_cols = len(cortex_columns)
        fig, axes = plt.subplots(
            1, num_cols, figsize=(5 * num_cols, 4),
            subplot_kw={'polar': True}
        )

        if num_cols == 1:  # ensure axes is iterable
            axes = [axes]

        # --- Plot each cortex reference ---
        for ax, col_name in zip(axes, cortex_columns):
            reference_angle = reference_values[col_name].iloc[0]
            self._plot_cortex_alignment(ax, reference_angle_deg=reference_angle, angles_deg=angles_deg)
            ax.set_title(f"{col_name.replace('_', ' ').title()} "
                         f"({reference_angle}°) for mouse {mouse_name}")

        # if self.display_parameters["save"]:
        # save_path = self.display_parameters["image_saving_folder"]
        # saving_name = f"cortex_orientation_{self.radon.config.patch_size}_{mouse_name}"
        # plt.savefig(os.path.join(save_path, saving_name), bbox_inches='tight', pad_inches=0, dpi=300)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def _plot_cortex_alignment(ax, reference_angle_deg, angles_deg):
        """
                Plot cortical orientation alignment as a polar histogram.

                Parameters
                ----------
                ax : matplotlib.axes._subplots.PolarAxesSubplot
                    Polar axis to draw on.
                reference_angle_deg : float
                    Reference orientation in degrees.
                angles_deg : np.ndarray
                    Array of angles in degrees.
                """

        # --- Compute angular differences (-180, 180] ---
        diff_deg = (angles_deg - reference_angle_deg + 180) % 360 - 180

        # Clip to ±90° (upper semicircle only)
        diff_deg = np.clip(diff_deg, -90, 90)

        # Convert to radians, rotate so that 0° aligns vertically
        theta = np.deg2rad(diff_deg) + np.pi / 2

        # --- Histogram ---
        num_bins = 18
        counts, bins = np.histogram(theta, bins=np.linspace(np.pi / 2 - np.pi / 2,
                                                            np.pi / 2 + np.pi / 2,
                                                            num_bins))

        # Plot histogram bars
        ax.bar((bins[:-1] + bins[1:]) / 2, counts,
               width=np.diff(bins),
               align='center', alpha=0.6, edgecolor='black')

        # --- Polar formatting ---
        ax.set_theta_zero_location("E")  # 0° at East
        ax.set_theta_direction(1)  # counter-clockwise
        ax.set_thetamin(0)
        ax.set_thetamax(180)

        # --- Custom angle labels ---
        tick_positions = np.arange(0, 181, 10)
        tick_labels = []
        for pos in tick_positions:
            diff = 90 - pos
            if diff > 0:
                tick_labels.append(f"{-diff}°")  # negative
            elif diff < 0:
                tick_labels.append(f"+{-diff}°")  # positive
            else:
                tick_labels.append(f"{reference_angle_deg}°")  # center
        ax.set_thetagrids(tick_positions, tick_labels)

        # Style tick labels
        for lbl in ax.get_xticklabels():
            if lbl.get_text() == f"{reference_angle_deg}°":
                lbl.set_fontsize(12)
                lbl.set_color("black")
                lbl.set_fontweight("bold")
            else:
                lbl.set_fontsize(8)
                lbl.set_color("gray")

        # --- ρ-axis formatting (force integer ticks) ---
        ax.set_yticks(range(0, int(max(counts)) + 1, 5))
        ax.set_yticklabels([str(i) for i in range(0, int(max(counts)) + 1, 5)])

        # --- Reference line ---
        ax.plot([np.pi / 2, np.pi / 2], [0, max(counts)],
                color='red', lw=2, label='Reference')
        # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15))

    def _filter_results_by_mask(self, results_dict, mask_dict):
        """
        Filter results in the same dict structure, keeping only entries
        where (x_center, y_center) lies within the mask.

        Args:
            results_dict (dict): Original results dictionary.
            mask_dict (dict): Paths to the masks.

        Returns:
            dict: Filtered results dictionary with same structure.
        """

        filtered = copy.deepcopy(results_dict)  # keep same structure & metadata

        for group, group_data in filtered.items():
            for mouse, entry in group_data.items():
                results = entry.get("results", [])
                filtered_results = []
                mask_path = mask_dict.get(self.masked_region, {}).get(group, {}).get(mouse, None)
                if mask_path:
                    mask = self._open_mask(mask_path)
                    for r in results:
                        x, y = int(round(r["x_center"])), int(round(r["y_center"]))
                        if (0 <= y < mask.shape[1]) and (0 <= x < mask.shape[0]):
                            if mask[x, y]:  # mask is True / nonzero
                                filtered_results.append(r)

                    # overwrite only the results list
                    entry["results"] = filtered_results
                else:
                    print(f"No mask found for group {group} mouse {mouse}")

        return filtered

    def _import_orientation_values(self, cvs_path, mouse=None):

        if mouse is None:
            mouse = self.mouse
            print(f"[INFO] 'mouse' was not provided. Using default: {mouse}")

        cortex_orientation_df = pd.read_csv(cvs_path)
        mouse_row = cortex_orientation_df[cortex_orientation_df["mouse"] == mouse]

        return mouse_row

    @staticmethod
    def _open_mask(mask_path):
        """
                Open a mask image and return it as a boolean NumPy array.

                Args:
                    mask_path (Path): Path to the mask image file.

                Returns:
                    np.ndarray: 2D boolean array where True = inside mask, False = outside.
                """
        mask_img = Image.open(mask_path).convert("L")

        mask = np.array(mask_img)

        mask_bool = mask > 0
        return mask_bool

    @staticmethod
    def _count_points(data_dict: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
        """
        Count the number of points in each 'results' list
        for every group and mouse in the given results dictionary.

        Args:
            data_dict (dict): Nested dictionary of the form

        Returns:
            dict: Dictionary with the same group/mouse structure,
                  but values are counts of results instead of the full entries.
        """
        points_dict = {}
        for group_key, group_value in data_dict.items():
            single_point = {}
            for mouse_key, mouse_value in group_value.items():
                single_point[mouse_key] = len(mouse_value["results"])
            points_dict[group_key] = single_point
        return points_dict
