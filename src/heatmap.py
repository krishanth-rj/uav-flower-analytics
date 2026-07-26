import numpy as np

def generate_density_grid(
    spatial_data,
    image_width,
    image_height,
    grid_size=8,
):

    density_grid = np.zeros((grid_size, grid_size), dtype=int)

    for detection in spatial_data:
        if detection.get("class_name") == "bud":
            continue

        center_x = detection.get("center_x")
        center_y = detection.get("center_y")

        if center_x is None or center_y is None:
            continue

        col = int((center_x / image_width) * grid_size) if image_width else 0
        row = int((center_y / image_height) * grid_size) if image_height else 0

        row = max(0, min(row, grid_size - 1))
        col = max(0, min(col, grid_size - 1))

        density_grid[row, col] += 1

    return density_grid