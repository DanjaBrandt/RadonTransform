import pprint

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats
from matplotlib.cm import get_cmap
import numpy as np
from pathlib import Path
import json
from collections import defaultdict

from scipy.spatial import cKDTree
from collections import defaultdict, Counter
import re
import matplotlib.gridspec as gridspec

from src.radon_transform_algorithm.utils import functions, process_functions


def _format_axes(axes, x_ticks, x_labels, y_lim):
    for ax in axes.flat:
        if ax.get_visible():
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_labels)
            ax.set_ylim(0, y_lim)


class DisplayCoregisterdImages:
    BIN_CENTERS = np.arange(0.5, 10.5)  # class‑level constant

    def __init__(self, **kwargs):
        self.display_mode = kwargs.get("display_mode", "images")

        self.coregistered_channels = kwargs.get("coregistered_channels", None)
        self.visualize_channel = kwargs.get("visualize_channel", None)
        self.groups = kwargs.get("groups", None)
        self.mouse = kwargs.get("mouse", None)
        self.day = kwargs.get("day", None)
        self.color = kwargs.get("color", None)
        self.lines = kwargs.get("lines", False)

        self.group_map = None
        self.plot_height = 3
        self.plot_width = 4
        self.font_size = 8
        self.tick_size = 6
        self.color_map = get_cmap("tab20")
        self.channels_colormap = {
            "2PM": 'magenta',
            "SHG": 'green'
        }

    def display(self, results_dict: dict):
        """
        Displays results based on the chosen display_mode.

        :param results_dict: Dictionary containing result data
        :type results_dict: dict
        """
        mode_map = {
            "images": self._display_images,
            "polar_histogram": self._display_polar_histogram,
            "widths": self._display_widths_distribution,
            "local_alignment": self._display_local_alignment,
            "angle_differences": self._display_raw_angle_differences,
            "mean_angle_differences": self._display_mean_angle_differences,
            "angular_co_alignment": self._display_angular_co_alignment,
            "count_points": self._display_points
        }

        display_function = mode_map.get(self.display_mode)
        if display_function:
            display_function(results_dict)
        else:
            print(f"Display mode '{self.display_mode}' is not supported yet.")

    def _display_images(self, results_dict: dict):
        self._display_grid(results_dict, draw_func=self._draw_image)

    def _display_polar_histogram(self, results_dict: dict):
        self._display_grid(results_dict, draw_func=self._draw_polar_histogram)

    def _display_widths_distribution(self, results_dict: dict):
        self._display_grid(results_dict, draw_func=self._draw_widths_distribution)

    def _display_local_alignment(self, results_dict: dict):
        self._display_grid(results_dict, draw_func=self._compute_local_alignment)

    def _display_raw_angle_differences(self, results_dict: dict):
        self._display_grid(results_dict, draw_func=self._compute_raw_angle_differences)

    def _display_mean_angle_differences(self, results_dict: dict):
        self._display_mean_grid(results_dict, draw_func=self._plot_mean_angle_differences)

    def _display_angular_co_alignment(self, results_dict: dict):
        # self._display_alignment_grid(results_dict, draw_func=self._plot_co_alignment) #other plot looks better
        self._display_day_grid(results_dict, draw_func=self._plot_co_alignment)

    def _display_points(self, results_dict: dict):
        self._display_day_grid(results_dict, draw_func=self._plot_points)

    def _draw_image(self, fig, ax_position, channel, day_str, mouse_name, results_dict):
        ax = fig.add_subplot(ax_position)
        colormap_dict = {}
        if self.color:
            colormap_dict = {
                "2PM": LinearSegmentedColormap.from_list("black_magenta", ["black", "magenta"]),
                "SHG": LinearSegmentedColormap.from_list("black_green", ["black", "green"])
            }
        input_data_name = results_dict[channel][mouse_name][day_str].get("input_data_name")
        image_path = Path("01_data") / channel / mouse_name / day_str / input_data_name
        image_array = process_functions.import_data(image_path)
        ax.imshow(image_array[0], cmap=colormap_dict.get(channel, "gray"))
        ax.axis("off")
        if self.lines:
            results_day = results_dict[channel][mouse_name][day_str]
            for idx, point in enumerate(results_day["results"]):
                center = (point["x_center"], point["y_center"])
                angle_radians = point["angle_radians"]
                width = point["width"]

                # Compute endpoints for the detected line (blue)
                start, end = functions.compute_endpoints(center, 10, angle_radians)
                ax.plot([start[1], end[1]], [start[0], end[0]], linewidth=2, color='b')

                # Mark the detected center with a small circle
                ax.plot(center[1], center[0], marker='o', markersize=2, color='cyan')
                ax.text(center[1] + 3, center[0] + 3, str(idx), color='cyan', fontsize=12)

                # Compute endpoints for the width indicator (red, perpendicular to main line)
                wstart, wend = functions.compute_endpoints(center, int(width), angle_radians + np.pi / 2)
                ax.plot([wstart[1], wend[1]], [wstart[0], wend[0]], linewidth=1, color='r')
        return ax

    def _draw_polar_histogram(self, fig, ax_position, channel, day_str, mouse_name, results_dict):
        """
        Draw a polar histogram for a given channel, day, and mouse using dynamic colormap.

        Parameters:
        - fig: matplotlib.figure.Figure
        - ax_position: subplot position or GridSpec cell
        - channel: str, channel name
        - day_str: str, day identifier
        - mouse_name: str, mouse identifier
        - results_dict: dict containing the data
        """
        colormap_dict = {}
        if self.color:
            colormap_dict = {
                "2PM": 'magenta',
                "SHG": 'green'
            }
        ax = fig.add_subplot(ax_position, projection='polar')

        num_bins = 10  # Keep the number of bins inside
        bins = np.linspace(0, 180, num_bins + 1)

        results_day = results_dict[channel][mouse_name][day_str]
        angles_deg = [entry['angle_degree'] for entry in results_day['results']]

        if len(angles_deg) == 0:
            # empty plot
            ax.set_theta_zero_location("E")
            ax.set_theta_direction(1)
            ax.set_thetamin(0)
            ax.set_thetamax(180)
            ax.set_yticks(np.arange(0, 25, 5))
            ax.tick_params(axis='both', which='major', labelsize=6)
            return ax

        counts, _ = np.histogram(angles_deg, bins=bins)

        # Convert bins to radians
        bin_edges_rad = np.radians(bins)
        bin_centers_rad = bin_edges_rad[:-1] + np.diff(bin_edges_rad) / 2

        # Draw bars with dynamic colormap per channel
        ax.bar(bin_centers_rad, counts,
               width=np.diff(bin_edges_rad),
               bottom=0,
               color=colormap_dict.get(channel, "gray"),
               edgecolor="gray",
               alpha=0.6,
               linewidth=1)

        # Polar formatting
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)
        ax.set_thetamin(0)
        ax.set_thetamax(180)
        ax.set_xticks(np.radians(np.linspace(0, 180, 7)))
        # ax.set_yticks([])
        maxcount = counts.max()
        ax.set_yticks(np.arange(0, maxcount + 3, 6, dtype=int))
        ax.tick_params(axis='both', which='major', labelsize=6, pad=0)

        return ax

    def _draw_widths_distribution(self, fig, ax_position, channel, day_str, mouse_name, results_dict):
        ax = fig.add_subplot(ax_position)
        colormap_dict = {
            "2PM": 'sturges',
            "SHG": 'sturges'}
        if self.color:
            colormap_dict = {
                "2PM": 'magenta',
                "SHG": 'green'
            }

        results_day = results_dict[channel][mouse_name][day_str]
        widths = [entry['width'] for entry in results_day['results']]

        # ax.hist(widths, bins='sturges', edgecolor='black')  # 'auto' chooses the best bin size
        counts, bins = np.histogram(widths, bins='sturges')
        ax.bar(bins[:-1], counts,
               width=np.diff(bins),
               # bottom=0,
               color=colormap_dict.get(channel, "gray"),
               align='edge',
               edgecolor="gray",
               alpha=0.6,
               linewidth=1)
        ax.set_ylim(bottom=0)
        ax.tick_params(axis='both', which='major', labelsize=6, pad=0)

        return ax

    def _compute_local_alignment(self, fig, ax_position, channel, day_str, mouse_name, results_dict):
        """
                 Computes local alignment between two structures from result dicts.
                 Display result as colorcoded points on the image.
                 Now supports result lists with dicts containing 'x_center', 'y_center', 'angle_degree'.
                 """
        source_key = channel
        target_key = [x for x in self.coregistered_channels if x != channel][0]

        input_data_name = results_dict[channel][mouse_name][day_str].get("input_data_name")
        image_path = Path("01_data") / channel / mouse_name / day_str / input_data_name
        source_image = process_functions.import_data(image_path)

        source_results = results_dict[source_key][mouse_name][day_str]
        source_coords, source_angles = self._extract_points(source_results['results'])

        alignment_results = {}

        target_results = results_dict[target_key][mouse_name][day_str]
        target_coords, target_angles = self._extract_points(target_results['results'])
        if len(target_coords) > 0:
            tree = cKDTree(target_coords)
            scores, inds = self._compute_alignment_scores(
                source_coords, source_angles, target_coords, target_angles, tree)
            alignment_results[target_key] = scores

        ax = fig.add_subplot(ax_position)
        self._display_alignment_map(source_image[0], ax, source_coords, alignment_results)
        return ax

    def _compute_raw_angle_differences(self, fig, ax_position, channel, day_str, mouse_name, results_dict):
        """
            Compute and plot a histogram of raw angle differences between two channels for a given mouse and day.

            This function:
            - Extracts coordinates and angles from the source and target channels.
            - Builds a KD-tree for the target points to compute nearest neighbor angle differences.
            - Plots a normalized histogram of the raw angle differences on the given figure and axes position.

            Parameters:
            ----------
            fig : matplotlib.figure.Figure
                The figure on which the histogram will be plotted.
            ax_position : int or tuple
                Position of the subplot in the figure (as in fig.add_subplot).
            channel : str
                The source channel name.
            day_str : str
                The day identifier (e.g., "day_07").
            mouse_name : str
                Name of the mouse.
            results_dict : dict
                Nested dictionary containing results for all channels, mice, and days.
                Expected structure: results_dict[channel][mouse][day]['results']


            Returns:
            -------
            ax : matplotlib.axes.Axes
                The axes object containing the histogram plot.
            """
        source_key = channel
        target_key = [x for x in self.coregistered_channels if x != channel][0]

        source_results = results_dict[source_key][mouse_name][day_str]
        source_coords, source_angles = self._extract_points(source_results['results'])

        target_results = results_dict[target_key][mouse_name][day_str]
        target_coords, target_angles = self._extract_points(target_results['results'])

        hist, bins, raw_diffs = self._compute_angle_diff_histogram(
            source_coords, source_angles, target_coords, target_angles, radius=30, bins=9, hist_range=(0, 90)
        )

        ax = fig.add_subplot(ax_position)
        normalized_hist_percent = (hist / hist.sum()) * 100
        ax.bar(bins[:-1], normalized_hist_percent, width=np.diff(bins), align='edge', color='skyblue',
               edgecolor='black')
        ax.set_xlabel("Angle Difference (degrees)")
        ax.set_title('\n test')
        ax.tick_params(axis='both', which='major', labelsize=self.tick_size, pad=0)
        ax.set_xlim(0, 90)
        # ax.set_ylim(0, 100)
        ax.grid(True)

        return ax

    def _compute_all_angle_differences_dict(self, results_dict, mice_dict, channel, day_name):
        """
        Compute angle difference histograms for a given set of mice on a specific day.

        This function compares angle differences between a source channel and its
        co-registered target channel. For each mouse in `mice_dict`, it extracts
        the coordinates and angles from both channels, computes the nearest-neighbor
        angle differences, and stores a normalized histogram of these differences.

        Parameters
        ----------
        results_dict : dict
            Nested results dictionary with structure:
            results_dict[channel][mouse][day]['results'].
        mice_dict : dict
            Dictionary mapping mouse identifiers to their available channels and days.
            Example structure:
                {
                    "m07_sl": {"2PM": ["day_07", "day_10"]},
                    "m12_sl": {"2PM": ["day_07", "day_10"]}
                }
        channel : str
            The source channel name (e.g., "2PM").
        day_name : str
            Specific day identifier (e.g., "day_07") to process.

        Returns
        -------
        dict
            Dictionary with structure:
                output[mouse] = histogram (np.ndarray)
            where each histogram contains the normalized angle difference distribution
            for that mouse on the given day.
        """
        source_key = channel
        # pick the other channel as target
        target_key = [x for x in self.coregistered_channels if x != channel][0]

        output_dict_hist_values = self._nested_dict()

        for mouse_key, mouse_value in mice_dict.items():
            source_results = results_dict[source_key][mouse_key][day_name]
            source_coords, source_angles = self._extract_points(source_results['results'])
            target_results = results_dict[target_key][mouse_key][day_name]
            target_coords, target_angles = self._extract_points(target_results['results'])
            # compute histogram
            hist, bins, raw_diffs = self._compute_angle_diff_histogram(
                source_coords, source_angles,
                target_coords, target_angles,
                radius=30, input_bins=10, hist_range=(0, 100))
            output_dict_hist_values[mouse_key] = hist

        return self._to_dict(output_dict_hist_values)

    def _plot_mean_angle_differences(self, fig, ax_position, results_dict, cmap):

        ax = fig.add_subplot(ax_position)
        valid_histograms = []
        for mouse_key, hist_values in results_dict.items():
            if hist_values is None:  # skip missing data
                continue
            x_positions = range(len(hist_values))  # bin indices
            y_positions = hist_values
            valid_histograms.append(hist_values)

            ax.scatter(
                x_positions,
                y_positions,
                color=cmap[mouse_key],
                label=mouse_key,
                s=20,
                alpha=0.6
            )
        if valid_histograms:
            stacked = np.vstack(valid_histograms)  # shape: (n_mice, n_bins)
            mean_per_bin = stacked.mean(axis=0)

            ax.scatter(
                np.arange(len(mean_per_bin)),
                mean_per_bin,
                color="red",
                marker="x",
                s=50,
                label="Mean"
            )
        n_bins = len(next(v for v in results_dict.values() if v is not None))
        ax.set_xticks(np.arange(n_bins))
        ax.set_xticklabels(np.arange(n_bins) * 10)
        ax.tick_params(axis='both', which='major', labelsize=self.tick_size, pad=0)

        return ax

    @staticmethod
    def _plot_co_alignment2(fig, ax_position, results_dict, mice_color):
        """
        Plot co-alignment ratios for multiple mice in a single subplot.

        This function works similarly to `_plot_co_alignment` but is designed to be
        used together with `_display_alignment_grid`. It creates a plot with two rows
        and three subplots for each day.

        Notes:
            - The axes of the subplots are independent (not shared).
            - This version improves plotting clarity but may require further adjustments
              for aesthetics or precision.

        Args:
            fig (matplotlib.figure.Figure): The figure object to plot on.
            ax_position (matplotlib.gridspec.SubplotSpec): The position of the subplot
                in the figure or gridspec.
            results_dict (dict): Dictionary containing histogram/angle data for each mouse.
            mice_color (dict): Dictionary mapping mouse keys to colors for plotting.

        Returns:
            matplotlib.axes.Axes: The Axes object of the created subplot.
        """
        ax = fig.add_subplot(ax_position)

        ratios_dict = {}

        for mouse_key, hist_values in results_dict.items():
            if hist_values is None:  # handle None case
                ratios_dict[mouse_key] = None
            else:
                denominator = sum(hist_values[:3])
                numerator = sum(hist_values[5:9])

                if denominator != 0:
                    ratio = numerator / denominator
                else:
                    ratio = None  # or 0.0 if you prefer
                ratios_dict[mouse_key] = ratio

        filtered = {k: v for k, v in ratios_dict.items() if v is not None}

        keys = list(filtered.keys())
        values = list(filtered.values())
        ax.boxplot(values, positions=[1], widths=0.3, whis=[0, 100])

        # --- Scatter (colored by mouse) ---
        for i, (mouse, val) in enumerate(filtered.items()):
            x_jittered = 1 + np.random.normal(0, 0.05)  # jitter around 1
            ax.scatter(x_jittered, val, color=mice_color[mouse], s=50, label=mouse)

        # ax.set_ylim(0, 2)
        # ax.axis('off')
        return ax

    def _plot_co_alignment(self, fig, ax_position, results_dict, channels, group_key, mice, days, mice_color):
        """
            Plot co-alignment ratios for multiple mice, channels, and days in a single subplot.

            For each channel and day, the function computes the alignment ratios per mouse using
            `_compute_all_angle_differences_dict` and `_compute_ratios`, and then plots them as
            scatter points and/or boxplots using `_plot_ratios`.

            Args:
                fig (matplotlib.figure.Figure): The figure object to plot on.
                ax_position (matplotlib.gridspec.SubplotSpec): Position of the subplot in the figure or gridspec.
                results_dict (dict): Nested dictionary containing alignment results per channel, mouse, and day.
                channels (list): List of channels to include in the plot.
                group_key (str): Name of the group (used in the subplot title).
                mice (dict): Dictionary of mice in this group.
                days (list): List of days to plot along the x-axis.
                mice_color (dict): Dictionary mapping each mouse to a color for plotting.

            Returns:
                matplotlib.axes.Axes: The axes object containing the plot.

            Notes:
                - The x-axis represents days.
                - Each day contains scatter/boxplot points for all mice.
                - The axes are styled using `self.font_size` and `self.tick_size`.
                - The group key in the title is automatically converted to uppercase.
            """
        ax = fig.add_subplot(ax_position)

        for c_idx, channel in enumerate(channels):
            for d_idx, day in enumerate(days):
                mean_results = self._compute_all_angle_differences_dict(results_dict, mice, channel, day)
                ratios_dict = self._compute_ratios(mean_results)
                self._plot_ratios(ax, ratios_dict, d_idx, mice_color)
        ax.set_xticks(range(len(days)))
        ax.set_xticklabels(days, fontsize=self.font_size)
        ax.tick_params(axis='y', which='major', labelsize=self.tick_size)
        ax.set_title(f'Alignment Ratios {group_key.upper()} Group', fontsize=self.font_size)
        return ax

    def _plot_points(self, fig, ax_position, results_dict, channels, group_key, mice, days, mice_color):
        """
        Plot count of points for individual mice and overlay mean counts per channel.

        Each day is plotted along the x-axis. Each channel's mean count is shown as a line
        connecting the daily averages. Individual mouse points are plotted with their
        corresponding color.

        Args:
            fig (Figure): Matplotlib figure object.
            ax_position (SubplotSpec): Subplot position from gridspec.
            results_dict (dict): Nested dictionary with results per channel/mouse/day.
            channels (list): List of channels to plot.
            group_key (str): Name of the group (used in title).
            mice (dict): Dictionary of mice in this group.
            days (list): List of days to plot.
            mice_color (dict): Dictionary mapping mice to colors.

        Returns:
            ax (Axes): Matplotlib axes object with the plot.
        """
        ax = fig.add_subplot(ax_position)
        mean_counts = {channel: [] for channel in channels}

        for d_idx, day in enumerate(days):
            mice_counts = {channel: {} for channel in channels}
            for c_idx, channel in enumerate(channels):
                for mouse_key, _ in mice.items():
                    results = results_dict[channel][mouse_key][day]
                    points, _ = self._extract_points(results['results'])
                    mice_counts[channel][mouse_key] = len(points)
                counts = list(mice_counts[channel].values())
                mean_counts[channel].append(sum(counts) / len(counts) if counts else 0)
            self._plot_scatter_points(ax, mice_counts, d_idx, mice_color)

        for channel, mean_vals in mean_counts.items():
            ax.plot(range(len(days)), mean_vals, color=self.channels_colormap[channel], linestyle='--', marker='x',
                    markersize=7, markeredgecolor='black', label=f'{channel}')
        ax.legend(loc='upper left', fontsize=self.tick_size)

        ax.set_xticks(range(len(days)))
        ax.set_xticklabels(days, fontsize=self.font_size)
        ax.tick_params(axis='y', which='major', labelsize=self.tick_size)
        ax.set_title(f'Point Counts {group_key.upper()} Group', fontsize=self.font_size)
        return ax

    def _plot_scatter_points(self, ax, points_dict, day_idx, mice_color):
        """Plot boxplot + scatter for one day."""

        # Scatter colored by mouse
        for channel_key, channel_val in points_dict.items():
            for mouse, val in channel_val.items():
                x_jittered = day_idx + np.random.normal(0, 0.05)
                ax.scatter(x_jittered, val,
                           facecolor=self.channels_colormap[channel_key],
                           s=30,
                           edgecolor=mice_color[mouse],
                           #label=mouse
                           )

    @staticmethod
    def _plot_ratios(ax, ratios_dict, day_idx, mice_color):
        """Plot boxplot + scatter for one day."""
        filtered = {k: v for k, v in ratios_dict.items() if v is not None}
        values = list(filtered.values())

        # Boxplot
        ax.boxplot(values, positions=[day_idx], widths=0.3, whis=[0, 100], flierprops=dict(marker=''))

        # Scatter colored by mouse
        for mouse, val in filtered.items():
            x_jittered = day_idx + np.random.normal(0, 0.05)
            ax.scatter(x_jittered, val, color=mice_color[mouse], s=50, label=mouse)

    @staticmethod
    def _compute_ratios(mean_results):
        """Compute numerator/denominator ratios for each mouse."""
        ratios_dict = {}
        for mouse_key, hist_values in mean_results.items():
            if hist_values is None:
                ratios_dict[mouse_key] = None
            else:
                denominator = sum(hist_values[:3])
                numerator = sum(hist_values[5:9])
                ratio = numerator / denominator if denominator != 0 else None
                ratios_dict[mouse_key] = ratio
        return ratios_dict

    @staticmethod
    def _display_alignment_map(image, ax, coords, alignment_scores):
        """
        Display points color-coded by local alignment.
        """
        ax.imshow(image, cmap='gray', alpha=0.5)
        ax.axis('off')

        for score_key, score_items in alignment_scores.items():
            alignment_scores = np.array(score_items)
            valid_mask = ~np.isnan(score_items)
            nan_mask = np.isnan(score_items)

            vmin, vmax = 0, 90
            ticks = np.arange(vmin, vmax + 1, 10)

            # Plot valid scores
            scatter = ax.scatter(coords[valid_mask, 1], coords[valid_mask, 0],
                                 c=alignment_scores[valid_mask],
                                 cmap='coolwarm', s=30, edgecolors='k',
                                 vmin=vmin,
                                 vmax=vmax
                                 )

            # Plot NaN scores as unfilled circles
            ax.scatter(coords[nan_mask, 1], coords[nan_mask, 0],
                       facecolors='none', edgecolors='k', s=30, label='NaN')

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = plt.colorbar(scatter, cax=cax, ticks=ticks)
            cbar.set_label('Local Alignment (°)', fontsize=8)
            cbar.ax.tick_params(labelsize=8)

            # ax.set_title(f'{self.acquisition_mode[0]} Points Colored by Local {score_key} Alignment ')

    def _display_grid(self, results_dict: dict, draw_func):

        display_grid_dict = self._create_display_dict(results_dict)

        # Filter groups
        if self.groups is not None:
            display_grid_dict = {g: mice for g, mice in display_grid_dict.items() if g in self.groups}
        # Filter mice
        if self.mouse is not None:
            display_grid_dict = {
                g: {m: times for m, times in mice.items() if m in self.mouse}
                for g, mice in display_grid_dict.items()
            }
        # Filter channels
        if self.visualize_channel is not None:
            for g, mice in display_grid_dict.items():
                for m, acq_modes in mice.items():
                    display_grid_dict[g][m] = {k: v for k, v in acq_modes.items() if k in self.visualize_channel}

        # Filter days
        if self.day is not None:
            display_grid_dict = {
                g: {
                    m: {tp: [day for day in day_list if day in self.day]
                        for tp, day_list in times.items()}
                    for m, times in mice.items()
                }
                for g, mice in display_grid_dict.items()
            }

        n_groups = len(display_grid_dict)

        if n_groups == 0:
            print("No matching groups to display.")
            return

        if self.mouse is not None:
            self.plot_height = len(self.mouse) + 1
        if self.day is not None:
            self.plot_width = len(self.day) + 1

        fig = plt.figure(figsize=(self.plot_width * n_groups, self.plot_height))
        outer_grid = gridspec.GridSpec(1, n_groups, figure=fig)

        for group_idx, (group_key, mice) in enumerate(display_grid_dict.items()):
            inner_grid = gridspec.GridSpecFromSubplotSpec(
                len(mice), 1, subplot_spec=outer_grid[group_idx], hspace=0.1
            )
            for mouse_idx, (mouse_name, channel_dict) in enumerate(mice.items()):
                n_days = len(next(iter(channel_dict.values())))
                n_cols = n_days * len(channel_dict)
                gs_mouse = gridspec.GridSpecFromSubplotSpec(
                    1, n_cols, subplot_spec=inner_grid[mouse_idx], wspace=0.1
                )

                for d_idx in range(n_days):
                    for c_idx, (channel, days_list) in enumerate(channel_dict.items()):
                        day_str = days_list[d_idx]
                        ax_idx = d_idx * len(channel_dict) + c_idx
                        sub_ax_position = gs_mouse[ax_idx]

                        # Call the external drawing function
                        ax = draw_func(fig, sub_ax_position, channel, day_str, mouse_name, results_dict)

                        # Optional titles / labels
                        if mouse_idx == 0:
                            ax.set_title(f"{day_str} - {channel}", fontsize=self.font_size)
                        if ax_idx == 0:
                            ax.text(
                                -0.1, 0.5, mouse_name, rotation=90,
                                fontsize=self.font_size, ha='center', va='center', transform=ax.transAxes
                            )

        plt.tight_layout()
        plt.show()

    def _display_mean_grid(self, results_dict: dict, draw_func):
        """
        Display a grid of per-mouse angle difference histograms and their means.

        The grid layout is:
            - Rows: groups
            - Columns: channels
            - Inner columns: days within each channel

        Each subplot shows:
            - Individual mouse bin values (scatter points)
            - Mean bin values across all mice (red cross)

        Supports filtering of:
            - Groups (self.groups)
            - Channels (self.visualize_channel: which channel take as reference to compute the difference)
            - Days (self.day)

        Parameters
        ----------
        results_dict : dict
            Nested dictionary of results per channel, mouse, and day.
        draw_func : callable
            Function to draw histograms/scatter plots on an Axes. It must accept:
                (fig, ax_position, mean_results, mice_colour_map)
        """

        display_grid_dict = self._create_display_dict(results_dict)
        # Filter groups
        if self.groups is not None:
            display_grid_dict = {g: mice for g, mice in display_grid_dict.items() if g in self.groups}

        # Filter channels
        if self.visualize_channel is not None:
            for g, mice in display_grid_dict.items():
                for m, acq_modes in mice.items():
                    display_grid_dict[g][m] = {k: v for k, v in acq_modes.items() if k in self.visualize_channel}

        # Filter days
        if self.day is not None:
            display_grid_dict = {
                g: {
                    m: {tp: [day for day in day_list if day in self.day]
                        for tp, day_list in times.items()}
                    for m, times in mice.items()
                }
                for g, mice in display_grid_dict.items()
            }

        n_groups = len(display_grid_dict)

        if n_groups == 0:
            print("No matching groups to display.")
            return

        channels = list(next(iter(next(iter(display_grid_dict.values())).values())).keys())
        n_channels = len(channels)

        # Use one mouse entry to extract days (all share same days here)
        sample_group = next(iter(display_grid_dict.values()))
        sample_mouse = next(iter(sample_group.values()))
        days = next(iter(sample_mouse.values()))
        n_days = len(days)

        fig = plt.figure(figsize=(self.plot_width * n_days * n_channels,
                                  self.plot_height * n_groups))
        outer_grid = gridspec.GridSpec(n_groups, n_channels, figure=fig)
        mice_colour_map = self._build_mouse_color_map(display_grid_dict)

        for g_idx, (group_key, mice) in enumerate(display_grid_dict.items()):
            for c_idx, channel in enumerate(channels):
                fig.text(0.5, 0.98, f"Reference channel: {channel}", ha="center", va="center", fontsize=10)
                inner_grid = gridspec.GridSpecFromSubplotSpec(
                    1, n_days, subplot_spec=outer_grid[g_idx, c_idx], wspace=0.2
                )

                for d_idx, day in enumerate(days):
                    ax_position = inner_grid[d_idx]
                    mean_results = self._compute_all_angle_differences_dict(results_dict, mice, channel, day)
                    ax = draw_func(fig, ax_position, mean_results, mice_colour_map)

                    # Titles and labels
                    if g_idx == 0:
                        ax.set_title(f"{day}", fontsize=self.font_size)
                    if d_idx == 0:
                        ax.text(
                            -0.1, 0.5, group_key, rotation=0,
                            fontsize=self.font_size, ha='center', va='center', transform=ax.transAxes)

        plt.tight_layout()
        plt.show()

    def _display_alignment_grid(self, results_dict: dict, draw_func):
        """
        Display a grid of co-alignment plots for multiple groups, mice, channels, and days.

        This function organizes the visualization of alignment results into a grid format,
        where each row corresponds to a group, each column corresponds to a channel, and
        subplots within a column represent different days. The plotting of individual
        subplots is delegated to a provided drawing function (`draw_func`).

        Features:
            - Filters groups, channels, and days based on user-defined settings
              (`self.groups`, `self.visualize_channel`, `self.day`).
            - Automatically builds a color map for each mouse.
            - Creates a figure with a grid layout using `matplotlib.gridspec`.
            - Optionally adds titles for days and labels for groups.
            - Can overlay reference lines or additional annotations across all subplots
              (currently commented out).

        Args:
            results_dict (dict): Dictionary containing alignment data for each group, mouse,
                channel, and day.
            draw_func (callable): Function that draws the actual plot on a given subplot.
                It should accept `(fig, ax_position, mean_results, mice_color)` as arguments.

        Notes:
            - The function dynamically adapts the number of rows, columns, and subplot
              layout based on the provided data.
            - The axes of the subplots are independent; shared axes can be added by
              uncommenting the `shared_ax` code.
            - The function currently prints the number of groups and other debugging info.

        Returns:
            None. Displays the figure with the full alignment grid.
        """

        display_grid_dict = self._create_display_dict(results_dict)
        # Filter groups
        if self.groups is not None:
            display_grid_dict = {g: mice for g, mice in display_grid_dict.items() if g in self.groups}

        # Filter channels
        if self.visualize_channel is not None:
            for g, mice in display_grid_dict.items():
                for m, acq_modes in mice.items():
                    display_grid_dict[g][m] = {k: v for k, v in acq_modes.items() if k in self.visualize_channel}

        # Filter days
        if self.day is not None:
            display_grid_dict = {
                g: {
                    m: {tp: [day for day in day_list if day in self.day]
                        for tp, day_list in times.items()}
                    for m, times in mice.items()
                }
                for g, mice in display_grid_dict.items()
            }

        n_groups = len(display_grid_dict)

        if n_groups == 0:
            print("No matching groups to display.")
            return

        channels = list(next(iter(next(iter(display_grid_dict.values())).values())).keys())
        n_channels = len(channels)

        # Use one mouse entry to extract days (all share same days here)
        sample_group = next(iter(display_grid_dict.values()))

        print(n_groups)
        sample_mouse = next(iter(sample_group.values()))
        days = next(iter(sample_mouse.values()))
        n_days = len(days)

        fig = plt.figure(figsize=(self.plot_width * n_days * n_channels,
                                  self.plot_height * n_groups))
        outer_grid = gridspec.GridSpec(n_groups, n_channels, figure=fig)
        mice_colour_map = self._build_mouse_color_map(display_grid_dict)

        for g_idx, (group_key, mice) in enumerate(display_grid_dict.items()):
            for c_idx, channel in enumerate(channels):
                fig.text(0.5, 0.98, f"Reference channel: {channel}", ha="center", va="center", fontsize=10)

                # Create inner grid: 1 row, n_days columns
                inner_grid = gridspec.GridSpecFromSubplotSpec(
                    1, n_days, subplot_spec=outer_grid[g_idx, c_idx], wspace=0.2
                )

                axes = []
                for d_idx, day in enumerate(days):
                    ax_position = inner_grid[d_idx]
                    # print(day)
                    mean_results = self._compute_all_angle_differences_dict(results_dict, mice, channel, day)
                    # pprint.pprint(mean_results)
                    # Draw on each day’s subplot
                    ax = draw_func(fig, ax_position, mean_results, mice_colour_map)
                    axes.append(ax)
                    if g_idx == 0:
                        ax.set_title(f"{day}", fontsize=self.font_size)
                    if d_idx == 0:
                        ax.text(
                            -0.1, 0.5, group_key, rotation=0,
                            fontsize=self.font_size, ha='center', va='center', transform=ax.transAxes)

                # shared_ax = fig.add_subplot(inner_grid[:, :], frame_on=False)  # spans all inner subplots
                # shared_ax.set_xticks([])  # hide x-axis ticks
                # shared_ax.set_yticks([])  # hide y-axis ticks
                # shared_ax.set_ylim(0, 2)
                # shared_ax.set_ylabel("Angle Difference (degrees)", labelpad=40)
                # shared_ax.set_xlabel("Bins / Days", labelpad=20)

                # Example: draw horizontal line at y=mean across all subplots
                mean_all = np.mean([ax.get_lines()[0].get_ydata() for ax in axes if ax.get_lines()])
                # shared_ax.axhline(y=0, color='black', linestyle='-', linewidth=2)
                # shared_ax.axvline(x=0, color='black', linestyle='-', linewidth=2)
        plt.tight_layout()
        plt.show()

    def _display_day_grid(self, results_dict: dict, draw_func):

        display_grid_dict = self._create_display_dict(results_dict)
        # Filter groups
        if self.groups is not None:
            display_grid_dict = {g: mice for g, mice in display_grid_dict.items() if g in self.groups}

        # Filter channels
        if self.visualize_channel is not None:
            for g, mice in display_grid_dict.items():
                for m, acq_modes in mice.items():
                    display_grid_dict[g][m] = {k: v for k, v in acq_modes.items() if k in self.visualize_channel}

        # Filter days
        if self.day is not None:
            display_grid_dict = {
                g: {
                    m: {tp: [day for day in day_list if day in self.day]
                        for tp, day_list in times.items()}
                    for m, times in mice.items()
                }
                for g, mice in display_grid_dict.items()
            }

        n_groups = len(display_grid_dict)

        if n_groups == 0:
            print("No matching groups to display.")
            return

        channels = list(next(iter(next(iter(display_grid_dict.values())).values())).keys())
        n_channels = len(channels)

        # Use one mouse entry to extract days (all share same days here)
        sample_group = next(iter(display_grid_dict.values()))

        sample_mouse = next(iter(sample_group.values()))
        days = next(iter(sample_mouse.values()))
        n_days = len(days)

        fig = plt.figure(figsize=(self.plot_width * n_groups,
                                  self.plot_height))
        outer_grid = gridspec.GridSpec(1, n_groups, figure=fig)
        mice_colour_map = self._build_mouse_color_map(display_grid_dict)

        for g_idx, (group_key, mice) in enumerate(display_grid_dict.items()):
            # Create inner grid: 1 row, n_days columns
            inner_grid = gridspec.GridSpecFromSubplotSpec(
                1, 1, subplot_spec=outer_grid[g_idx], wspace=0.2
            )
            ax_position = inner_grid[0, 0]
            ax = draw_func(fig, ax_position, results_dict, channels, group_key, mice, days, mice_colour_map)
        plt.tight_layout()
        plt.show()

    def _create_display_dict(self, results_dict: dict):

        output_grid = defaultdict(lambda: defaultdict(dict))

        for channel_key, channel_value in results_dict.items():
            for mouse_key, mouse_value in channel_value.items():
                mouse_group = mouse_key[-2:].lower()

                if self.groups is None:
                    group_key = "all"
                else:
                    if mouse_group not in self.groups:
                        continue
                    group_key = mouse_group
                output_grid[group_key][mouse_key][channel_key] = list(mouse_value.keys())

        return output_grid

    @staticmethod
    def _extract_points(result_list):
        coords = np.array([[r["x_center"], r["y_center"]] for r in result_list])
        angles = np.array([r["angle_degree"] for r in result_list])
        return coords, angles

    def _compute_alignment_scores(self, source_coords, source_angles, target_coords, target_angles, tree,
                                  radius, sigma, weighted=True, return_individual_diffs=False):
        scores = []
        all_inds = {}
        all_raw_diffs = []
        all_distances = []

        for idx, (s_coord, s_angle) in enumerate(zip(source_coords, source_angles)):
            indices = tree.query_ball_point(s_coord, radius)
            all_inds[idx] = indices

            if not indices:
                scores.append(np.nan)
                continue

            t_coords = target_coords[indices]
            t_angles = target_angles[indices]
            distances = np.linalg.norm(t_coords - s_coord, axis=1)
            angle_diffs = self._angle_diff(s_angle, t_angles)

            if weighted:
                weights = self._gaussian_weight(distances, sigma)
                mean_diff = np.average(angle_diffs, weights=weights)
            else:
                mean_diff = np.mean(angle_diffs)

            scores.append(mean_diff)

            if return_individual_diffs:
                all_raw_diffs.extend(angle_diffs.tolist())
                all_distances.extend(distances.tolist())

        if return_individual_diffs:
            return scores, all_inds, all_raw_diffs, all_distances

        return scores, all_inds

    def _angle_diff(self, a1, a2):
        diff = np.abs(a1 - a2) % 180
        return np.minimum(diff, 180 - diff)

    def _gaussian_weight(self, d, sigma):
        return np.exp(-0.5 * (d / sigma) ** 2)

    def _compute_angle_diff_histogram(self, source_coords, source_angles,
                                      target_coords, target_angles,
                                      radius=30, input_bins=10, hist_range=(0, 100)):
        """
        Compute normalized histogram of angle differences between source and target.

        Returns
        -------
        normalized_hist : np.ndarray
            Normalized histogram (probability distribution).
        bin_edges : np.ndarray
            Histogram bin edges.
        raw_diffs : np.ndarray
            Individual raw angle differences.
        """
        if source_coords is None or target_coords is None:
            return None, None, None
        if source_coords.size == 0 or target_coords.size == 0:
            return None, None, None

        tree = cKDTree(target_coords)
        _, _, raw_diffs, _ = self._compute_alignment_scores(
            source_coords,
            source_angles,
            target_coords,
            target_angles,
            tree,
            radius=radius,
            sigma=1,
            weighted=False,
            return_individual_diffs=True
        )

        raw_diffs = np.asarray(raw_diffs, dtype=float)
        hist, bin_edges = np.histogram(raw_diffs, bins=input_bins, range=hist_range)

        normalized_hist = hist / hist.sum() if hist.sum() > 0 else None

        return normalized_hist, bin_edges, raw_diffs

    def _nested_dict(self):
        """Utility function to create infinitely nested defaultdicts."""
        return defaultdict(self._nested_dict)

    def _to_dict(self, d):
        """Recursively convert nested defaultdicts to plain dicts."""
        if isinstance(d, defaultdict):
            return {k: self._to_dict(v) for k, v in d.items()}
        elif isinstance(d, dict):
            return {k: self._to_dict(v) for k, v in d.items()}
        else:
            return d

    def _build_mouse_color_map(self, display_grid_dict):
        """
        Build a mapping of mouse_key -> color.

        Parameters
        ----------
        display_grid_dict : dict
            Nested dict of structure {group: {mouse: {channel: [days...]}}}
        cmap_name : str
            Name of matplotlib colormap to use.

        Returns
        -------
        dict
            Dictionary mapping mouse_key -> color (RGBA tuple).
        """
        cmap = plt.get_cmap(self.color_map)

        # Collect all mouse keys across groups
        all_mouse_keys = [
            mouse_key
            for group_dict in display_grid_dict.values()
            for mouse_key in group_dict.keys()
        ]

        # Assign colors evenly spaced in the colormap
        colors = [cmap(i / max(1, len(all_mouse_keys) - 1)) for i in range(len(all_mouse_keys))]

        return dict(zip(all_mouse_keys, colors))
