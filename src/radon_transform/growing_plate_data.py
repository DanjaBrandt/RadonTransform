import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import seaborn as sns
import pandas as pd
from pathlib import Path

from utils import functions
from radon_analysis import RadonStructureDetection
from config import Config


class GrowingPlateAnalysis:
    def __init__(self, radon_analyzer, **kwargs):
        self.config = 6
        self.radon = radon_analyzer
        self.display_parameters = kwargs.get('display_parameters', None)
        # pprint.pprint(display_param)

    def run(self, data_folder):
        data_dict = bone_analysis.create_data_dict(data_folder)
        # self.display_rois_subplots(data_dict)
        # self.count_signal(data_dict)
        # signal_dict = self.analyze_signal_all(data_dict)
        # pprint.pprint(signal_dict)
        # self.display_signal_plot(signal_dict)

        # patch_array, metadata = self.create_subimages(data_dict)

        # pprint.pprint(metadata)

        # radon_results = self.run_radon_analysis(data_dict)
        # print(len(radon_results))
        # self.display_rois(data_dict=data_dict, results=radon_results)
        # self.display_cortex_alignment(radon_results)
        # points_dict = self.analyze_points_all(data_dict)
        # self.display_points_plot(points_dict)

        # SINGLE ROIS

        # pprint.pprint(data_dict)
        group = self.display_parameters["group"]
        mouse = self.display_parameters["mouse"]
        selected_rois = self.load_selected_rois(data_dict, group, mouse)
        result_selected_rois = self.run_radon_analysis_on_selected_rois(data_dict)
        self.display_selected_rois(data_dict=data_dict, specific_selection='bad', results=result_selected_rois)
        # roi = self._load_single_roi(self.display_parameters["specific_roi"])
        # plt.imshow(roi)
        # plt.show()

    def run_radon_analysis(self, data_dict, group=None, mouse=None):

        patch_array, metadata = self.create_subimages(data_dict, group=group, mouse=mouse)

        print(patch_array.shape)
        print(len(metadata))

        # pprint.pprint(metadata)
        # self.radon._background_threshold_value = 6
        # print( self.radon._background_threshold_value)
        all_results = []
        for idx, (patch, meta) in enumerate(zip(patch_array, metadata)):
            self.radon._background_threshold_value = int(meta["threshold"])
            _, result = self.radon.process(patch[np.newaxis, :])

            if len(result) > 0:

                y_shift = meta["col_start"]
                x_shift = meta["row_start"]
                # print(f"patch id {idx} len res: {len(result)} roi idx: {meta['roi_index']}")
                for res in result:
                    res["center"] = (res["center"][0] + x_shift, res["center"][1] + y_shift)
                    res["p_start"] = (res["p_start"][0] + x_shift, res["p_start"][1] + y_shift)
                    res["p_start"] = (res["p_end"][0] + x_shift, res["p_end"][1] + y_shift)
                    res["roi_index"] = meta["roi_index"]
                    res["group"] = meta["group"]
                    res["mouse"] = meta["mouse"]

                    # pprint.pprint(res)
                all_results.append(result)

                # x_roi_shift = x_shift + meta['subsize'] * meta['roi_index']

        print('len all results: ', len(all_results))
        '''
        grouped_patches = []
        positions = []

        for patch, meta in zip(patch_array, metadata):
            if meta["roi_index"] == 0:
                grouped_patches.append(patch)
                positions.append((meta["row_start"], meta["col_start"]))

        # Determine full canvas size
        max_row = max(r + p.shape[0] for p, (r, c) in zip(grouped_patches, positions))
        max_col = max(c + p.shape[1] for p, (r, c) in zip(grouped_patches, positions))

        # Assume patches are 2D (grayscale); change to (H, W, C) if RGB
        canvas = np.zeros((max_row, max_col), dtype=grouped_patches[0].dtype)

        # Place each patch into the canvas
        for patch, (r, c) in zip(grouped_patches, positions):
            h, w = patch.shape
            canvas[r:r + h, c:c + w] = patch  # Paste the patch into the canvas



        plt.imshow(canvas, cmap='gray')
        count = 0
        for res_id, patch_result in enumerate(all_results):
            for id, re in enumerate(patch_result):
                y, x = re["center"]
                plt.plot(x, y, 'ro')
                plt.annotate(f"{str(res_id)}-{str(id)}", (x, y), textcoords="offset points", xytext=(5, 5), ha='center', color='white')

                count = count + 1
                angle_radians = re["angle_radians"]
                width = re["width"]
                #print(f"res id: {res_id} x,y: {x}, {y}, angle: {angle_radians}, width: {width}")
                # Compute endpoints for the detected line (blue)
                start, end = functions.compute_endpoints(re["center"], 10, angle_radians)
                plt.plot([start[1], end[1]], [start[0], end[0]], linewidth=2, color='b')
                wstart, wend = functions.compute_endpoints(re["center"], int(width), angle_radians + np.pi / 2)
                plt.plot([wstart[1], wend[1]], [wstart[0], wend[0]], linewidth=1, color='r')
        print('count', count)
        plt.show()
        '''
        return all_results
        # print(all_results)

    def run_radon_analysis_on_selected_rois(self, data_dict, group=None, mouse=None):
        if group is None or mouse is None:
            if group is None:
                group = self.display_parameters["group"]
                print(f"[INFO] 'group' was not provided. Using default: {group}")

            if mouse is None:
                mouse = self.display_parameters["mouse"]
                print(f"[INFO] 'mouse' was not provided. Using default: {mouse}")

        selected_rois = self.load_selected_rois(data_dict=data_dict, group=group, mouse=mouse)
        output = {
            'group': selected_rois['group'],
            'mouse': selected_rois['mouse'],
            'results': {}
        }
        for roi_group, roi_dict in selected_rois['selected_rois'].items():
            output['results'][roi_group] = {}
            for roi_key, arr in roi_dict.items():
                threshold = np.percentile(arr, 50)
                self.radon._background_threshold_value = int(threshold)
                _, result = self.radon.process(arr[np.newaxis, :])
                output['results'][roi_group][roi_key] = result

        return output

    def create_subimages(self, data_dict, subsize=200, group=None, mouse=None):
        result_list = self.analyze_rois(data_dict=data_dict, group=group, mouse=mouse)
        patches = []
        metadata = []  # To store info about each patch

        row_shift = 0
        for idx, resul in enumerate(result_list):
            roi = resul["roi_crop"]
            roi_2d = roi[:, :, 0]  # shape (H, W)

            h, w = roi_2d.shape
            for i in range(0, h - subsize + 1, subsize):
                for j in range(0, w - subsize + 1, subsize):
                    patch = roi_2d[i:i + subsize, j:j + subsize]
                    patches.append(patch)
                    metadata.append({
                        "roi_index": idx,  # which roi_crop (from result_list)
                        "roi_shape": resul["roi_shape"],
                        "subsize": subsize,
                        "group": resul["group"],  # optional: add group info
                        "mouse": resul["mouse"],  # optional: add mouse ID
                        "row_start": i + row_shift,
                        "col_start": j,
                        "height": h,
                        "width": w,
                        "threshold": resul["threshold"]
                    })
            row_shift = row_shift + resul["roi_shape"][0] - 1

        patch_array = np.array(patches)  # shape: (N_patches, subsize, subsize)
        print('Total ROI patches:', patch_array.shape)

        return patch_array, metadata

    def _check_patches_position(self, pat):

        layout = [
            (3, 21),  # 3 rows of 21
            (3, 21),  # 3 more rows of 21
            (2, 18),  # 2 rows of 18
            (1, 15),  # 1 row of 15
        ]

        fig, axs = plt.subplots(nrows=9, ncols=21, figsize=(21, 9))  # Make enough space
        patch_idx = 0

        # Flatten axs (2D) into 1D list for easier indexing
        axs = axs.reshape(-1)

        # Fill patches according to the layout
        for row_count, cols in layout:
            for r in range(row_count):
                for c in range(cols):
                    ax = axs[patch_idx]
                    ax.imshow(patch_array[patch_idx], cmap='gray')
                    ax.axis('off')
                    patch_idx += 1

        # Hide remaining axes (if any)
        for i in range(patch_idx, len(axs)):
            axs[i].set_visible(False)

        # plt.tight_layout()
        plt.show()

    def analyze_signal_all(self, data_dict):

        signal_dict = {}
        for group_key, group_value in data_dict.items():
            single_signal = {}
            for mouse_key, mouse_value in group_value.items():
                result_list = self.analyze_rois(data_dict=data_dict, group=group_key, mouse=mouse_key)
                single_signal[mouse_key] = self.count_signal(result_list)
            signal_dict[group_key] = single_signal
        # pprint.pprint(signal_dict)
        return signal_dict

    def analyze_points_all(self, data_dict):

        points_dict = {}
        for group_key, group_value in data_dict.items():
            single_point = {}
            for mouse_key, mouse_value in group_value.items():
                radon_results = self.run_radon_analysis(data_dict=data_dict, group=group_key, mouse=mouse_key)
                single_point[mouse_key] = len(radon_results)
            points_dict[group_key] = single_point
        # pprint.pprint(signal_dict)
        return points_dict

    def analyze_rois(self, data_dict, percentile=70, group=None, mouse=None):
        """
        Extracts ROIs from the image and computes stats:
        - Threshold (by percentile)
        - Total pixel count
        - Signal pixel count (above threshold)

        Returns a list of dicts, one per ROI.
        """
        if group is None or mouse is None:
            if group is None:
                group = self.display_parameters["group"]
                print(f"[INFO] 'group' was not provided. Using default: {group}")

            if mouse is None:
                mouse = self.display_parameters["mouse"]
                print(f"[INFO] 'mouse' was not provided. Using default: {mouse}")

        image_path = data_dict[group][mouse]["image"][0]
        zip_path = data_dict[group][mouse]["rois"]

        data = self.import_multichannel_image(image_path)
        roi_list = self.load_rois(zip_path)

        results = []
        for idx, roi in enumerate(roi_list):
            coords = roi['coordinates']
            xs, ys = coords[:, 0], coords[:, 1]
            x_min, x_max = int(xs.min()), int(xs.max()) + 1
            y_min, y_max = int(ys.min()), int(ys.max()) + 1

            roi_crop = data[0, y_min:y_max, x_min:x_max]
            # print('min', roi_crop.min())
            # print('max', roi_crop.max())
            threshold = np.percentile(roi_crop, percentile)
            # print(f"roi {idx}, threshold: {threshold}")
            total_pixels = roi_crop.size
            signal_pixels = np.sum(roi_crop > threshold)

            results.append({
                "group": group,
                "mouse": mouse,
                "index": idx,
                "roi_crop": roi_crop,
                "roi_shape": roi_crop.shape,
                "threshold": threshold,
                "total_pixels": total_pixels,
                "signal_pixels": signal_pixels,
                "fraction_above_threshold": signal_pixels / total_pixels
            })

        return results

    def count_signal(self, rois_results_list):
        # rois_results_list = self.analyze_rois(data_dict)

        total_pixels_all = sum(r["total_pixels"] for r in rois_results_list)
        signal_pixels_all = sum(r["signal_pixels"] for r in rois_results_list)

        overall_fraction = signal_pixels_all / total_pixels_all if total_pixels_all > 0 else 0

        # print(f"Overall signal pixels: {signal_pixels_all}")
        # print(f"Overall total pixels: {total_pixels_all}")
        # print(f"Overall fraction above threshold: {overall_fraction:.2%}")
        return {
            "signal_pixels": signal_pixels_all,
            "total_pixels": total_pixels_all,
            "overall_fraction": overall_fraction
        }

    def load_selected_rois(self, data_dict, group, mouse):

        image_path = Path(data_dict[group][mouse]["image"][0])

        folder_path = image_path.parent

        data = self.import_multichannel_image(image_path)
        data = data[0, :, :, 0]

        roi_types = ["border", "edge", "inner"]
        roi_dict = {rtype: {} for rtype in roi_types}

        for rtype in roi_types:
            roi_folder = folder_path / rtype
            for roi_file in roi_folder.glob("*.roi"):
                prefix = roi_file.stem.split("_")[0]
                single_roi = self._load_single_roi(roi_file)
                coords = single_roi["coordinates"]

                x_min, y_min = coords[:, 0].min(), coords[:, 1].min()
                x_max, y_max = coords[:, 0].max(), coords[:, 1].max()
                x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])

                subimg = data[y_min:y_max, x_min:x_max]
                roi_dict[rtype][prefix] = subimg

        return {
            'group': group,
            'mouse': mouse,
            'selected_rois': roi_dict
        }

    def display_selected_rois(self, data_dict, group=None, mouse=None, specific_selection=None, results=None):
        if group is None or mouse is None:
            if group is None:
                group = self.display_parameters["group"]
                print(f"[INFO] 'group' was not provided. Using default: {group}")

            if mouse is None:
                mouse = self.display_parameters["mouse"]
                print(f"[INFO] 'mouse' was not provided. Using default: {mouse}")
        selected_rois = self.load_selected_rois(data_dict, group, mouse)
        # Swap: rows = ROI types, columns = samples
        row_keys = [specific_selection] if specific_selection else list(
            next(iter(selected_rois["selected_rois"].values())).keys())
        col_keys = list(selected_rois["selected_rois"].keys())

        n_rows = len(row_keys)
        n_cols = len(col_keys)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2 * n_rows))

        # Ensure axes is always 2D
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = np.array([axes])
        elif n_cols == 1:
            axes = np.array([[ax] for ax in axes])

        for row_idx, row_key in enumerate(row_keys):
            for col_idx, col_key in enumerate(col_keys):
                ax = axes[row_idx, col_idx]
                img = selected_rois["selected_rois"][col_key][row_key]
                # threshold = np.percentile(img, 50)
                # binary_img = img > threshold

                ax.imshow(img, cmap="gray")
                ax.axis("off")
                if row_idx == 0:
                    ax.set_title(col_key, fontsize=10)

                if results:
                    result = results["results"][col_key][row_key]
                    for subresult in result:
                        angle_radians = subresult["angle_radians"]
                        width = subresult["width"]

                        center = subresult["center"]

                        start, end = functions.compute_endpoints(center, 10, angle_radians)
                        ax.plot([start[1], end[1]], [start[0], end[0]], linewidth=2, color='b')

                        # Mark the detected center with a small circle
                        ax.plot(center[1], center[0], marker='o', markersize=2, color='cyan')

                        # Compute endpoints for the width indicator (red, perpendicular to main line)
                        wstart, wend = functions.compute_endpoints(center, int(width), angle_radians + np.pi / 2)
                        ax.plot([wstart[1], wend[1]], [wstart[0], wend[0]], linewidth=1, color='r')
                if self.display_parameters["save"]:
                    save_path = self.display_parameters["image_saving_folder"]
                    saving_name = f"radon_selected_rois_{self.radon.config.patch_size}_{mouse}_{group}_{row_key}"
                    plt.savefig(os.path.join(save_path, saving_name), bbox_inches='tight', pad_inches=0, dpi=300)

            fig.text(0.08, 1 - (row_idx + 0.5) / n_rows, row_key, rotation=0, va='center', ha='right', fontsize=10)

        plt.tight_layout()
        plt.show()

    def display_rois(self, data_dict, results=None):

        group = self.display_parameters["group"]
        mouse = self.display_parameters["mouse"]

        image_path = data_dict[group][mouse]["image"][0]
        zip_path = data_dict[group][mouse]["rois"]

        data = self.import_multichannel_image(image_path)
        roi_list = self.load_rois(zip_path)

        fig, ax = plt.subplots()

        ax.imshow(data[0], cmap='gray')
        for roi in roi_list:
            coords = roi['coordinates']
            xs, ys = coords[:, 0], coords[:, 1]
            # print(xs, ys)
            ax.plot(xs, ys, label=roi['name'])

        ax.set_title(f'ROIs for {mouse} of {group} group')
        ax.set_xlabel(f"width: {data[0].shape[1]} pixels")
        ax.set_ylabel(f"heigh: {data[0].shape[0]} pixels")

        if self.display_parameters["save"]:
            save_path = self.display_parameters["image_saving_folder"]
            saving_name = f"rois_{mouse}_{group}"
            plt.savefig(os.path.join(save_path, saving_name), bbox_inches='tight', pad_inches=0, dpi=300)

        if results:
            for result in results:
                for subresult in result:
                    angle_radians = subresult["angle_radians"]
                    width = subresult["width"]

                    y, x = subresult["center"]
                    roi_index = subresult["roi_index"]

                    # Shift a single coordinate point from ROI
                    coords_0 = roi_list[0]['coordinates']
                    coords = roi_list[roi_index]['coordinates']
                    xs_0, ys_0 = coords_0[0]  # first point
                    xs, ys = coords[0]  # first point

                    new_center = (y + ys_0, x + xs)

                    # ax.plot(x + xs, y + ys_0, 'ro')

                    start, end = functions.compute_endpoints(new_center, 70, angle_radians)
                    ax.plot([start[1], end[1]], [start[0], end[0]], linewidth=2, color='b')

                    # Mark the detected center with a small circle
                    ax.plot(x + xs, y + ys_0, marker='o', markersize=2, color='cyan')

                    # Compute endpoints for the width indicator (red, perpendicular to main line)
                    wstart, wend = functions.compute_endpoints(new_center, int(width), angle_radians + np.pi / 2)
                    ax.plot([wstart[1], wend[1]], [wstart[0], wend[0]], linewidth=1, color='r')
        if self.display_parameters["save"]:
            save_path = self.display_parameters["image_saving_folder"]
            saving_name = f"radon_{self.radon.config.patch_size}_{mouse}_{group}"
            plt.savefig(os.path.join(save_path, saving_name), bbox_inches='tight', pad_inches=0, dpi=300)

        plt.show()

    def display_rois_subplots(self, data_dict):

        rois_results_list = self.analyze_rois(data_dict)

        n = len(rois_results_list)
        fig, axes = plt.subplots(n, 3, figsize=(10, 1.5 * n))

        if n == 1:
            axes = axes[np.newaxis, :]  # Ensure 2D indexing if only 1 ROI

        for idx, roi in enumerate(rois_results_list):
            group = roi["group"]
            mouse = roi["mouse"]
            roi_idx = roi["index"]
            roi_crop = roi["roi_crop"]
            threshold = roi["threshold"]
            total_pixels = roi["total_pixels"]
            signal_pixels = roi["signal_pixels"]
            fraction = roi["fraction_above_threshold"]

            binary_roi = (roi_crop > threshold).astype(int) * 255

            # Original ROI
            axes[idx, 0].imshow(roi_crop, cmap='gray')
            axes[idx, 0].set_title(f"ROI {roi_idx} - {group}: {mouse}")
            axes[idx, 0].axis('off')

            # Binarized ROI
            axes[idx, 1].imshow(binary_roi, cmap='gray')
            axes[idx, 1].set_title(f"ROI {roi_idx} - Binarized")
            axes[idx, 1].axis('off')

            # Text info
            text = (
                f"Threshold: {threshold:.1f}\n"
                f"Total px: {total_pixels}\n"
                f"Signal px: {signal_pixels}\n"
                f"Fraction: {fraction:.2%}"
            )
            axes[idx, 2].text(0.1, 0.5, text, fontsize=10, va='center')
            axes[idx, 2].axis('off')

        plt.tight_layout()
        if self.display_parameters["save"]:
            save_path = self.display_parameters["image_saving_folder"]
            saving_name = f"rois_analysis_{mouse}_{group}"
            plt.savefig(os.path.join(save_path, saving_name), bbox_inches='tight', pad_inches=0, dpi=300)

        plt.show()

    def display_signal_plot(self, signal_plot):
        # Flatten the data into a list of records
        records = []
        for group, subjects in signal_plot.items():
            for subject, metrics in subjects.items():
                records.append({
                    'group': group,
                    'subject': subject,
                    'total_pixels': metrics['total_pixels'],
                    'signal_pixels': metrics['signal_pixels'],
                    'overall_fraction': metrics['overall_fraction'] * 100
                })

        df = pd.DataFrame(records)

        palette = sns.color_palette("tab10", n_colors=df['subject'].nunique())

        # Ensure the group order is young, middle, old
        df['group'] = pd.Categorical(df['group'], categories=['young', 'middle', 'old'], ordered=True)

        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True)

        # --- Left plot: Total Pixels ---
        sns.boxplot(x='group', y='total_pixels', data=df, ax=axes[0], color='lightgray')
        sns.stripplot(x='group', y='total_pixels', data=df, hue='subject', ax=axes[0],
                      jitter=True, size=5, palette=palette, dodge=True, legend=False)
        axes[0].set_title('Total Pixels by Group')
        axes[0].set_ylabel("Total Pixels")

        # --- Center plot: Signal Pixels ---
        sns.boxplot(x='group', y='signal_pixels', data=df, ax=axes[1], color='lightgray')
        sns.stripplot(x='group', y='signal_pixels', data=df, hue='subject', ax=axes[1],
                      jitter=True, size=5, palette=palette, dodge=True, legend=False)
        axes[1].set_title('Signal Pixels by Group')
        axes[1].set_ylabel("Signal Pixels")

        # --- Right plot: Overall Fraction [%] ---
        sns.boxplot(x='group', y='overall_fraction', data=df, ax=axes[2], color='lightgray')
        sns.stripplot(x='group', y='overall_fraction', data=df, hue='subject', ax=axes[2],
                      jitter=True, size=5, palette=palette, dodge=True)
        axes[2].set_title('Overall Fraction by Group')
        axes[2].set_xlabel("Groups")
        axes[2].set_ylabel("Signal Pixels / Total Pixels %")

        axes[2].legend_.remove()
        # Move legend outside
        handles, labels = axes[2].get_legend_handles_labels()
        fig.legend(handles, labels, title='Subject', bbox_to_anchor=(0.95, 0.5), loc='right')

        # Adjust layout to make space for legend
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Make room on right

        if self.display_parameters["save"]:
            save_path = self.display_parameters["image_saving_folder"]
            saving_name = f"pixel_analysis_all"
            plt.savefig(os.path.join(save_path, saving_name), bbox_inches='tight', pad_inches=0, dpi=300)
        plt.show()

    def display_points_plot(self, signal_plot):
        # Flatten the data into a list of records
        records = []
        for group, subjects in signal_plot.items():
            for subject, metrics in subjects.items():
                records.append({
                    'group': group,
                    'subject': subject,
                    'detected_points': metrics
                })

        df = pd.DataFrame(records)

        palette = sns.color_palette("tab10", n_colors=df['subject'].nunique())

        # Ensure the group order is young, middle, old
        df['group'] = pd.Categorical(df['group'], categories=['young', 'middle', 'old'], ordered=True)

        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(6, 4), sharex=True)

        # --- Left plot: Total Pixels ---
        sns.boxplot(x='group', y='detected_points', data=df, ax=ax, color='lightgray')
        sns.stripplot(x='group', y='detected_points', data=df, hue='subject', ax=ax,
                      jitter=True, size=5, palette=palette, dodge=True, legend=False)
        ax.set_title('Detected Points by Group')
        ax.set_ylabel("Detected Points")

        plt.tight_layout()
        if self.display_parameters["save"]:
            save_path = self.display_parameters["image_saving_folder"]
            saving_name = f"points_analysis_{self.radon.config.patch_size}_all"
            plt.savefig(os.path.join(save_path, saving_name), bbox_inches='tight', pad_inches=0, dpi=300)
        plt.show()

    def plot_cortex_alignment(self, ax, reference_angle_deg, angles_deg):
        # Compute signed angular differences (-180 to 180)
        diff_deg = (angles_deg - reference_angle_deg + 180) % 360 - 180

        # Clip to ±90° to keep only upper semicircle
        diff_deg = np.clip(diff_deg, -90, 90)

        # Convert to radians, shift so 0° diff = vertical (π/2)
        theta = np.deg2rad(diff_deg) + np.pi / 2

        # Create bins
        num_bins = 18
        counts, bins = np.histogram(theta, bins=np.linspace(np.pi / 2 - np.pi / 2,
                                                            np.pi / 2 + np.pi / 2,
                                                            num_bins))

        # Plot histogram bars
        ax.bar((bins[:-1] + bins[1:]) / 2, counts,
               width=np.diff(bins),
               align='center', alpha=0.6, edgecolor='black')

        # Polar formatting
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)
        ax.set_thetamin(0)
        ax.set_thetamax(180)

        # Tick positions & labels
        tick_positions = np.arange(0, 181, 10)
        tick_labels = []
        for pos in tick_positions:
            diff = 90 - pos
            if diff > 0:
                tick_labels.append(f"{-diff}°")
            elif diff < 0:
                tick_labels.append(f"+{-diff}°")
            else:
                tick_labels.append(f"{reference_angle_deg}°")
        ax.set_thetagrids(tick_positions, tick_labels)

        # Style tick labels: center one large & black, others smaller & gray
        for lbl in ax.get_xticklabels():
            if lbl.get_text() == f"{reference_angle_deg}°":
                lbl.set_fontsize(12)
                lbl.set_color("black")
                lbl.set_fontweight("bold")
            else:
                lbl.set_fontsize(8)
                lbl.set_color("gray")

        # Reference line
        ax.plot([np.pi / 2, np.pi / 2], [0, max(counts)],
                color='red', lw=2, label='Reference')
        # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15))

    def display_cortex_alignment(self, results):
        reference_values = self.import_cortex_orientation_values(self.display_parameters["cortex_orientation_csv"])
        left_reference_angle = reference_values["left_cortex_dir"].iloc[0]
        right_reference_angle = reference_values["right_cortex_dir"].iloc[0]
        mouse_name = reference_values.loc[1, "mouse"]

        angles = []

        for sublist in results:
            for item in sublist:
                angles.append(float(item['angle_degree']))  # Convert to float if needed

        angles_deg = np.array(angles)

        # angles_deg = np.array([10, 10, 10, 40, 55, 60, 73, 80, 90,
        # 103, 120, 120, 120, 133, 150, 165, 170])

        fig, axes = plt.subplots(1, 2, figsize=(10, 4), subplot_kw={'polar': True})

        # Left reference
        self.plot_cortex_alignment(axes[0], reference_angle_deg=left_reference_angle, angles_deg=angles_deg)
        axes[0].set_title(f"Left Cortex Orientatoin ({left_reference_angle}°) for mouse {mouse_name}")

        # Right reference
        self.plot_cortex_alignment(axes[1], reference_angle_deg=right_reference_angle, angles_deg=angles_deg)
        axes[1].set_title(f"Right Cortex Orientation ({right_reference_angle}°) for mouse {mouse_name}")

        if self.display_parameters["save"]:
            save_path = self.display_parameters["image_saving_folder"]
            saving_name = f"cortex_orientation_{self.radon.config.patch_size}_{mouse_name}"
            plt.savefig(os.path.join(save_path, saving_name), bbox_inches='tight', pad_inches=0, dpi=300)

        plt.tight_layout()
        plt.show()

    def import_cortex_orientation_values(self, cvs_path, mouse=None):

        if mouse is None:
            mouse = self.display_parameters["mouse"]
            print(f"[INFO] 'mouse' was not provided. Using default: {mouse}")

        cortex_orientation_df = pd.read_csv(cvs_path)
        mouse_row = cortex_orientation_df[cortex_orientation_df["mouse"] == mouse]

        return mouse_row

    def load_rois(self, roi_path: str) -> list:
        """
        Load one or more ROIs from a .zip file or a single .roi file.
        Returns a list of dicts with 'name', 'coordinates', and 'roi'.
        """
        roi_list = []

        if roi_path.lower().endswith('.zip'):
            roi_list.extend(self._load_zip_rois(roi_path))

        elif roi_path.lower().endswith('.roi'):
            roi_list.append(self._load_single_roi(roi_path))

        else:
            raise ValueError("Unsupported ROI file type: must be .roi or .zip")

        return roi_list

    def _load_zip_rois(self, zip_path: str) -> list:
        import zipfile
        import roifile

        rois = []
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for name in zf.namelist():
                if name.lower().endswith('.roi'):
                    roi_data = zf.read(name)
                    roi = roifile.ImagejRoi.frombytes(roi_data)
                    coords = roi.coordinates()
                    if coords is not None:
                        rois.append({
                            'name': name.replace('.roi', ''),
                            'coordinates': coords,
                            'roi': roi
                        })
        return rois

    def _load_single_roi(self, roi_path: str) -> dict:
        import roifile
        import os

        with open(roi_path, 'rb') as f:
            roi_data = f.read()
            roi = roifile.ImagejRoi.frombytes(roi_data)
            coords = roi.coordinates()
            if coords is None:
                raise ValueError(f"No coordinates found in ROI: {roi_path}")
            return {
                'name': os.path.splitext(os.path.basename(roi_path))[0],
                'coordinates': coords,
                'roi': roi
            }

    @staticmethod
    def import_multichannel_image(path: str, normalize: bool = True, max_value: int = 255,
                                  min_value: int = 0, single_channel: bool = True) -> np.ndarray:
        with Image.open(path) as img:
            h, w = img.size[1], img.size[0]  # height, width
            n_frames = getattr(img, 'n_frames', 1)  # handle single frame images gracefully

            # If you want to keep channels, get count, else assume grayscale
            c = len(img.getbands()) if not single_channel else 1

            # Initialize array: frames, height, width, channels (1 if grayscale)
            tiffarray = np.zeros((n_frames, h, w, c), dtype=np.float32)

            for i in range(n_frames):
                img.seek(i)
                imarray = np.array(img)  # shape: (h, w, c) or (h, w)

                # If grayscale but 3 channels, extract first channel
                if single_channel and imarray.ndim == 3:
                    imarray = imarray[:, :, 0]

                # Normalize if requested
                if normalize:
                    min_val = imarray.min()
                    max_val = imarray.max()
                    # Avoid division by zero
                    if max_val > min_val:
                        norm = (imarray - min_val) * ((max_value - min_value) / (max_val - min_val)) + min_value
                    else:
                        norm = imarray  # or zeros_like(imarray)
                    imarray = norm

                # If single_channel, expand dims to keep consistent shape
                if single_channel:
                    imarray = np.expand_dims(imarray, axis=-1)

                tiffarray[i] = imarray

        return tiffarray

    @staticmethod
    def create_data_dict(root_folder: str) -> dict:
        """
        Creates a nested dictionary from a folder containing 'young', 'middle', 'old',
        where each subfolder (e.g. 'y1', 'y2') contains ROI zip files and image files.
        """
        roi_ext = ('.zip', '.roi')
        image_exts = ('.tif', '.tiff', '.png', '.jpg')

        data_dict = {
            group: {
                subfolder: {
                    'rois': next(
                        (os.path.join(subfolder_path, f) for f in os.listdir(subfolder_path)
                         if f.lower().endswith(roi_ext) and f.lower().startswith("roiset")),
                        None
                    ),
                    'image': [
                        os.path.join(subfolder_path, f)
                        for f in os.listdir(subfolder_path)
                        if f.lower().endswith(image_exts)
                    ]
                }
                for subfolder in os.listdir(group_path)
                if os.path.isdir(subfolder_path := os.path.join(group_path, subfolder))
            }
            for group in os.listdir(root_folder)
            if os.path.isdir(group_path := os.path.join(root_folder, group))
        }

        return data_dict


if __name__ == "__main__":
    data_folder = "C:/Users/Danja Brandt/Desktop/FU/SideProjects/George/inner_data"

    display_param = {
        'group': 'young',
        'mouse': 'y2',
        'image_saving_folder': "C:/Users/Danja Brandt/Desktop/FU/SideProjects/George/images/preliminary_images",
        'cortex_orientation_csv': "C:/Users/Danja Brandt/Desktop/FU/SideProjects/George/cortex_dir.csv",
        'specific_roi': "C:/Users/Danja Brandt/Desktop/FU/SideProjects/George/selected_rois/young/y2/good_in.roi",
        # selected_rois
        'save': 1
    }
    radon_param = {
        "patch_size": 20,
        "patch_step": 10,
        "sigma": 0,
    }
    input_config = Config(**radon_param)
    radon_class = RadonStructureDetection(input_config)

    bone_analysis = GrowingPlateAnalysis(radon_analyzer=radon_class, display_parameters=display_param)

    bone_analysis.run(data_folder)

    # pprint.pprint(data_dict)
