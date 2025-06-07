import xml.etree.ElementTree as ET
import gzip
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd
import numpy as np
import glob
import argparse
import warnings


def get_lr_count(points_plant):
    lr_count = len(points_plant[points_plant["root_type"] == "lateral"])
    return lr_count


def get_angles(points_all):
    # Step 1: Dynamically determine the canvas size
    x_vals, y_vals = zip(*points_all)
    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals), max(y_vals)

    # Add some padding
    padding = 10
    width = x_max - x_min + 2 * padding
    height = y_max - y_min + 2 * padding

    # Create a blank image (size based on bounding box)
    img = np.zeros((height, width), dtype=np.uint8)

    # Step 2: Draw points on the image (shift coordinates to fit)
    for x, y in points_all:
        img[y - y_min + padding, x - x_min + padding] = 255  # Adjust position

    # Step 3: Apply Probabilistic Hough Transform
    lines = cv2.HoughLinesP(
        img, rho=1, theta=np.pi / 180, threshold=10, minLineLength=10, maxLineGap=5
    )

    # get angle of each line
    angles = []  # Store angles

    if lines is not None:
        # Calculate angles
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Avoid division by zero (vertical line case)
            if x2 - x1 == 0:
                angle_with_vertical = 0  # Perfectly vertical line
            else:
                theta = np.arctan2(y2 - y1, x2 - x1)  # Angle with horizontal
                theta_deg = np.degrees(theta)  # Convert to degrees
                angle_with_vertical = 90 - abs(theta_deg)  # Angle with vertical

            angles.append(angle_with_vertical)
        angles = [float(angle) for angle in angles]
    return angles


def get_all_points(points_plant):
    all_points = [point for sublist in points_plant["points"] for point in sublist]
    # all_points = [(int(x), int(y)) for sublist in points_plant['points'] for x, y in sublist]
    return all_points


def get_width_height(all_points):
    left = min(x for x, y in all_points)
    right = max(x for x, y in all_points)
    top = min(y for x, y in all_points)
    bottom = max(y for x, y in all_points)
    width = abs(right - left)
    height = abs(bottom - top)
    return width, height


def get_pr(points_plant):
    pr_points = points_plant[points_plant["root_type"] == "primary"]
    pr_points = [point for sublist in points_plant["points"] for point in sublist]
    return pr_points


def get_lr(points_plant):
    lr_points = points_plant[points_plant["root_type"] == "lateral"]
    lr_points = [point for sublist in lr_points["points"] for point in sublist]
    return lr_points


def get_length(points):
    # Convert to NumPy array
    points_array = np.array(points)
    # Calculate Euclidean distances between consecutive points
    distances = np.sqrt(np.sum(np.diff(points_array, axis=0) ** 2, axis=1))
    # Sum up all distances
    line_length = np.sum(distances)
    return line_length


def get_lengths(points_plant):
    pr_points = points_plant[points_plant["root_type"] == "primary"]["points"]
    lr_points = points_plant[points_plant["root_type"] == "lateral"]["points"]

    # get primary root line length
    pr_lengths = []
    for pr_point in pr_points:
        pr_length = get_length(pr_point)
        pr_lengths.append(pr_length)

    # get lateral root line length
    lr_lengths = []
    for lr_point in lr_points:
        lr_length = get_length(lr_point)
        lr_lengths.append(lr_length)

    pr_length = np.sum(pr_lengths)
    lr_length = np.sum(lr_lengths)
    tr_length = pr_length + lr_length
    return pr_length, lr_length, tr_length


def get_tip_depths(points_plant, pr_points, all_points):
    # get root system top location
    top = min(y for x, y in all_points)
    # get pr tip
    pr_bottom = max(y for x, y in pr_points)
    pr_depth = float(np.nanmean(pr_bottom - top))

    # get lr tip
    lr_points = points_plant[points_plant["root_type"] == "lateral"]
    lr_bottoms = [max(y for x, y in points) for points in lr_points["points"]]
    lr_depth = float(np.nanmean([b - top for b in lr_bottoms]))

    # get tip depths
    bottoms = np.append(lr_bottoms, pr_bottom)
    depth = float(np.nanmean([b - top for b in bottoms]))

    return pr_depth, lr_depth, depth


def get_convhull_area(all_points):
    hull = cv2.convexHull(np.array(all_points))
    area = cv2.contourArea(hull)
    return area


def get_sdx_sdy(all_points):
    sdx = np.std(np.array(all_points)[:, 0])
    sdy = np.std(np.array(all_points)[:, 1])
    sdx_sdy_ratio = sdx / sdy
    return sdx, sdy, sdx_sdy_ratio


def get_trace_info(trace_files, scale):
    traits = pd.DataFrame()
    for trace_file in trace_files:
        print(f"trace_file: {trace_file}")
        with gzip.open(trace_file, "rt", encoding="utf-8", errors="ignore") as file:
            tree = ET.parse(file)
            root = tree.getroot()
        trace_infromation = ET.tostring(root, encoding="utf-8").decode("utf-8")

        # Extract required path attributes
        path_data = []
        for path in root.findall(".//path"):  # Find all 'path' elements
            path_info = {
                "id": path.get("id"),
                "name": path.get("name"),
                "startson": path.get("startson"),
                "startx": path.get("startx"),  # Might be None if missing
                "startsindex": path.get("startsindex"),  # Might be None if missing
            }
            path_data.append(path_info)
        path_data = pd.DataFrame(path_data)

        # Extract points from paths
        all_points = []
        for path in root.findall("path"):
            points = [
                (int(point.attrib["x"]), int(point.attrib["y"]))
                for point in path.findall("point")
            ]
            all_points.append(points)

        # # Create 'plantid' by counting NaN occurrences in 'startx'
        # path_data["plantid"] = path_data["startx"].isna().cumsum()
        # Identify primary roots (where startson is NaN)
        path_data["plantid"] = None  # Initialize plantid column
        path_data.loc[path_data["startson"].isna(), "plantid"] = path_data[
            "id"
        ]  # Assign primary root id

        # Map lateral roots to the nearest primary root
        path_data["plantid"] = (
            path_data["plantid"].ffill().astype("Int64")
        )  # Forward fill plant IDs

        # Rename plantid to 1, 2, 3, ...
        unique_ids = path_data["plantid"].unique()  # Get unique plant IDs
        id_mapping = {
            old_id: new_id for new_id, old_id in enumerate(unique_ids, 1)
        }  # Create a mapping
        path_data["plantid"] = path_data["plantid"].map(id_mapping)  # Apply the new IDs

        path_data["image_name"] = trace_file

        # save the plantid, root type (primary or lateral), path points
        points_plant_df = pd.DataFrame()
        for i, trace in enumerate(all_points):
            # Convert to float if possible, else assume it's not NaN
            value = path_data["startsindex"][i]
            if value is None or pd.isna(value):  # Check if None or NaN-compatible
                is_nan = True
            else:
                try:
                    is_nan = np.isnan(float(value))  # Convert to float and check
                except (ValueError, TypeError):
                    is_nan = False  # If conversion fails, it's not NaN
            root_type = "primary" if is_nan else "lateral"
            plantid = path_data["plantid"][i]
            points = all_points[i]
            path = pd.DataFrame(
                [
                    {
                        "image_name": trace_file,
                        "plantid": plantid,
                        "root_type": root_type,
                        "points": points,
                    }
                ]
            )
            points_plant_df = pd.concat([points_plant_df, path])
        points_plant_df = points_plant_df.reset_index().drop(columns=["index"])

        # for each plant
        unique_plants = np.unique(points_plant_df["plantid"])

        for plant in unique_plants:
            points_plant = points_plant_df[points_plant_df["plantid"] == plant]

            lr_count = get_lr_count(points_plant)

            all_points = get_all_points(points_plant)
            angles_avg = np.nanmean(get_angles(all_points))

            width, height = get_width_height(all_points)

            pr_points = get_pr(points_plant)
            lr_points = get_lr(points_plant)
            pr_length, lr_length, tr_length = get_lengths(points_plant)

            pr_depth, lr_depth, depth = get_tip_depths(
                points_plant, pr_points, all_points
            )

            sdx, sdy, sdx_sdy_ratio = get_sdx_sdy(all_points)

            convhull_area = get_convhull_area(all_points)

            # convert to cm with scale
            width_cm = width / scale
            height_cm = height / scale
            tr_length_cm = tr_length / scale
            pr_length_cm = pr_length / scale
            lr_length_cm = lr_length / scale
            depth_cm = depth / scale
            pr_depth_cm = pr_depth / scale
            lr_depth_cm = lr_depth / scale
            convhull_area_cm2 = convhull_area / (scale * scale)

            traits_plant = pd.DataFrame(
                [
                    {
                        "image name": trace_file,
                        "plantid": plant,
                        "width (pixel)": width,
                        "width (cm)": width_cm,
                        "height (pixel)": height,
                        "height (cm)": height_cm,
                        "number of lr": lr_count,
                        "total root length (pixel)": tr_length,
                        "total root length (cm)": tr_length_cm,
                        "pr length (pixel)": pr_length,
                        "pr length (cm)": pr_length_cm,
                        "lr length (pixel)": lr_length,
                        "lr length (cm)": lr_length_cm,
                        "tip depth (pixel)": depth,
                        "tip depth (cm)": depth_cm,
                        "pr tip depth (pixel)": pr_depth,
                        "pr tip depth (cm)": pr_depth_cm,
                        "lr tip depth (pixel)": lr_depth,
                        "lr tip depth (cm)": lr_depth_cm,
                        "sdx": sdx,
                        "sdy": sdy,
                        "sdx/sdy": sdx_sdy_ratio,
                        "convex hull area (pixel)": convhull_area,
                        "convex hull area (cm2)": convhull_area_cm2,
                        "average angle": angles_avg,
                    }
                ]
            )
            traits = pd.concat([traits, traits_plant])

    return traits


def main():
    parser = argparse.ArgumentParser(description="Mimulus phenotyping based on traces")
    parser.add_argument("--trace_folder", required=True, help="tarce path")
    parser.add_argument("--save_folder", required=True, help="Save path")

    args = parser.parse_args()

    trace_root_folder = args.trace_folder
    save_folder = args.save_folder

    # suppress warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # get subfolders
    trace_folders = [
        file
        for file in os.listdir(trace_root_folder)
        if os.path.isdir(os.path.join(trace_root_folder, file))
    ]

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    scale = 314  # 1 cm = 314 pixels, 12 cms = 3768 pixels

    for trace_folder in trace_folders:
        trace_files = glob.glob(
            os.path.join(
                os.path.join(trace_root_folder, trace_folder), "**", "*.traces"
            ),
            recursive=True,
        )
        traits = get_trace_info(trace_files, scale)
        # traits = get_trace_info(trace_files)
        traits["image name"] = traits["image name"].str.replace(
            trace_folder + "/", "", regex=False
        )
        traits.to_csv(
            os.path.join(save_folder, f"{trace_folder}_traits.csv"), index=False
        )

    # connect all traits files
    traits_files = glob.glob(os.path.join(save_folder, "*_traits.csv"))
    all_traits = pd.DataFrame()
    for traits_file in traits_files:
        traits = pd.read_csv(traits_file)
        all_traits = pd.concat([all_traits, traits])
    all_traits.to_csv(os.path.join(save_folder, "all_traits.csv"), index=False)


if __name__ == "__main__":
    main()
