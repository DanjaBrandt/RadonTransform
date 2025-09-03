import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import copy
import seaborn as sns
import pandas as pd
from pathlib import Path
import pprint
from typing import Dict, Any

from src.radon_transform_algorithm.utils import functions, process_functions


class DisplayGrowthPlate:
    def __init__(self, **kwargs):
        self.display_mode = kwargs.get("display_mode", "images")

        self.group = kwargs.get("group", None)
        self.mouse = kwargs.get("mouse", None)
        self.masked_region = kwargs.get("masked_region", None)
        self.alignment = kwargs.get("alignment", None)

        self.save = kwargs.get("save", False)
        self.result_nr = Path(kwargs.get("result_folder")).name.split("_")[-1]
        self.csv_path = "C:/Users/Danja Brandt/Desktop/FU/SideProjects/George/plate_cortex_dir.csv"

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
            "angle_deviation": self._display_angle_std
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
        if self.masked_region:
            image_name = f"{self.display_mode}_{self.masked_region}_{self.group}_{self.mouse}_{self.result_nr}.svg"

            plt.imshow(self.debug_mask, cmap='gray', alpha=0.5)
        else:
            image_name = f"{self.display_mode}_{self.group}_{self.mouse}_{self.result_nr}.svg"

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
        if self.save:
            folder_path = Path('../02_growing_plate/02_images') / f"output_{self.result_nr}" / self.group
            saving_path = Path(folder_path) / image_name
            plt.savefig(saving_path, format="svg")

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
            ax=ax, color="lightgray", whis=[0, 100], width=0.4,
            boxprops=dict(alpha=0.5)
        )
        sns.stripplot(
            x="group", y="detected_points", data=df,
            hue="subject", ax=ax, jitter=True, size=5,
            palette=palette, dodge=True, legend=False
        )

        ax.set_title("Detected Points by Group")
        ax.set_xlabel("Groups")
        ax.set_ylabel("Detected Points")

        plt.tight_layout()
        if self.save:
            image_name = (
                f"{self.display_mode}_{self.masked_region}_{self.result_nr}.svg"
                if self.masked_region else
                f"{self.display_mode}_{self.result_nr}.svg"
            )

            saving_path = Path('../02_growing_plate/02_images') / f"output_{self.result_nr}" / image_name
            plt.savefig(saving_path, format="svg")

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
        reference_values = self._import_orientation_values(self.csv_path)

        if reference_values.empty:
            raise ValueError("Reference values CSV is empty")

        # --- Get mouse name ---
        try:
            mouse_name = reference_values.iloc[0]["mouse"]
        except KeyError:
            raise ValueError("CSV must contain a 'mouse' column")

        # --- Extract all cortex direction columns ---
        # print(reference_values.columns)
        cortex_columns = [col for col in reference_values.columns if "_dir" in col]
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

        if self.save:
            image_name = (
                f"{self.display_mode}_{self.masked_region}_{self.group}_{self.mouse}_{self.result_nr}.svg"
                if self.masked_region else
                f"{self.display_mode}_{self.group}_{self.mouse}_{self.result_nr}.svg"
            )

            folder_path = Path('../02_growing_plate/02_images') / f"output_{self.result_nr}" / self.group
            saving_path = Path(folder_path) / image_name
            plt.savefig(saving_path, format="svg")

        plt.tight_layout()
        plt.show()

    def _display_angle_std(self, results_dict):
        """
        Display boxplot of the std of cortex orientation alignment for a given mouse.

        Parameters
        ----------
        results_dict : dict
            Dictionary containing angle results for groups and mice.
        """

        # reference_values = self._import_orientation_values(csv_path)

        # --- Get mouse name ---
        # mouse_name = reference_values.iloc[0]["mouse"]
        # print(mouse_name)
        std_dict = {}
        for group_key, group_value in results_dict.items():
            mouse_dict = {}
            for mouse_key, mouse_value in group_value.items():
                std = self._compute_std(results=mouse_value['results'], mouse_key=mouse_key)
                # stds.append(std)
                mouse_dict[mouse_key] = std
            std_dict[group_key] = mouse_dict

        records = [
            {"group": group, "subject": subject, "stds": count}
            for group, subjects in std_dict.items()
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
            x="group", y="stds", data=df,
            ax=ax, color="lightgray", whis=[0, 100], width=0.4,
            boxprops=dict(alpha=0.5)
        )
        sns.stripplot(
            x="group", y="stds", data=df,
            hue="subject", ax=ax, jitter=True, size=5,
            palette=palette, dodge=False, legend=False
        )

        ax.set_title(f"Stds by Group. Reference structure: {self.alignment}; region: {self.masked_region}")
        ax.set_xlabel("Groups")
        ax.set_ylabel("Standard Deviations Points")

        if self.save:
            image_name = (
                f"{self.display_mode}_ref-{self.alignment}_reg-{self.masked_region}_{self.result_nr}.svg"
                if self.masked_region else
                f"{self.display_mode}_ref-{self.alignment}_all_{self.result_nr}.svg"
            )

            folder_path = Path('../02_growing_plate/02_images') / f"output_{self.result_nr}"
            saving_path = Path(folder_path) / image_name
            plt.savefig(saving_path, format="svg")

        plt.tight_layout()

    def _compute_std(self, results, mouse_key):
        if self.alignment is None:
            raise ValueError("Specify reference strcture for alignment")
        reference_structure = f"{self.alignment}_dir"
        angles = [float(item['angle_degree']) for item in results]
        angles_deg = np.array(angles)
        reference_values = self._import_orientation_values(cvs_path=self.csv_path, mouse=mouse_key)
        reference_angle = reference_values[reference_structure].iloc[0]
        diff_deg = self._compute_angle_differences(angles_deg=angles_deg, reference_angle_deg=reference_angle)
        #plt.hist(diff_deg, bins=18, edgecolor='black')
        #plt.xlabel("Value")
        #plt.ylabel("Frequency")
        #plt.title(f"Angle diff for {mouse_key} with ref angle {reference_angle}")
        #plt.show()
        std = np.std(diff_deg)
        return std

    def _plot_cortex_alignment(self, ax, reference_angle_deg, angles_deg):
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

        # --- Compute angular differences, lines can be perpendicular or parallel ---

        # diff_deg = np.abs((angles_deg - reference_angle_deg) % 180)
        # diff_deg = np.where(diff_deg > 90, 180 - diff_deg, diff_deg)
        diff_deg = self._compute_angle_differences(angles_deg=angles_deg, reference_angle_deg=reference_angle_deg)

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
        raw_interval = int(np.ceil(max(counts) / 8))
        tick_interval = int(np.ceil(raw_interval / 10.0) * 10)

        ax.set_yticks(range(0, int(max(counts)) + 1, tick_interval))
        ax.set_yticklabels([str(i) for i in range(0, int(max(counts)) + 1, tick_interval)])

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
        if self.masked_region:
            debug_mask_path = mask_dict.get(self.masked_region).get(self.group).get(self.mouse)
            self.debug_mask = self._open_mask(debug_mask_path)

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
    def _compute_alignment_difference(angles_deg, reference_angle_deg):
        diff_deg = np.abs((angles_deg - reference_angle_deg) % 180)
        diff_deg = np.where(diff_deg > 90, 180 - diff_deg, diff_deg)
        return diff_deg

    @staticmethod
    def _compute_angle_differences(angles_deg, reference_angle_deg):
        """
        Compute signed angular differences when angles are restricted to [0, 180].
        Result is in [-90, 90], preserving left/right of reference.
        """
        diff_deg = (angles_deg - reference_angle_deg + 90) % 180 - 90
        #print('diff', angles_deg - reference_angle_deg + 90)

        return np.round(diff_deg, 2)

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
