# UAV Flower Analytics

UAV Flower Analytics is a research-oriented Python project for segmenting
flowers and buds in aerial imagery, summarising detections, and mapping flower
density across an image. It uses an Ultralytics YOLO segmentation model trained
on two classes:

- `bud`
- `flower`

The repository is structured so that prediction, analysis, spatial extraction,
density calculation, and visualization can be reused independently.

## Quick start

This project requires Python 3.10 or newer.

```bash
git clone <repository-url>
cd uav-flower-analytics
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the complete workflow on an image:

```bash
python3 src/demo.py data/raw/test_images/garden.jpg
```

This writes the annotated prediction using the input filename and saves the
density heatmap to `results/density_heatmap.png`. The command also prints a
summary of flower, bud, and total-detection counts.

## Model and data requirements

Inference expects the trained weights at:

```text
runs/segment/models/training/flower_segmentation/weights/best.pt
```

The dataset and model artifacts are intentionally ignored by Git because they
are large. Obtain or generate them before running inference. The expected
dataset layout is compatible with Ultralytics segmentation:

```text
data/raw/flower_dataset/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

Labels are YOLO segmentation polygons with normalized coordinates. The
included `data.yaml` names the classes `bud` and `flower`.

## Project workflow

```text
input image
    │
    ▼
predict.py ──► Ultralytics Results
                  │
       ┌──────────┴──────────┐
       ▼                     ▼
 analyze.py              spatial.py
       │                     │
       ▼                     ▼
 summary statistics     per-object positions
                              │
                              ▼
                         heatmap.py
                              │
                              ▼
                         visualize.py
```

The `demo.py` script orchestrates this workflow without duplicating the logic
in these modules.

## Modules

| Module | Purpose |
| --- | --- |
| `src/predict.py` | Loads the trained YOLO model, validates an image path, runs segmentation, and saves an annotated prediction. |
| `src/analyze.py` | Produces detection counts, per-class confidence values and averages, and original image dimensions from a `Results` object. |
| `src/spatial.py` | Converts every detection into class, confidence, original-pixel box geometry, and mask polygon data. |
| `src/heatmap.py` | Bins spatial detections into a density grid. Buds are excluded from the flower-density grid. |
| `src/visualize.py` | Renders a supplied density matrix with Matplotlib. |
| `src/test_heatmap.py` | Runs inference and prints an 8×8 density grid for one image. |
| `src/demo.py` | Runs the complete prediction, analysis, density, and heatmap workflow. |

## Common commands

Run only segmentation inference and save the annotated image:

```bash
python3 src/predict.py data/raw/test_images/garden.jpg
```

Print the 8×8 density grid without rendering a figure:

```bash
python3 src/test_heatmap.py data/raw/test_images/garden.jpg
```

Run the same scripts as Python modules if preferred:

```bash
python3 -m src.demo data/raw/test_images/garden.jpg
```

## Reusing the modules

The core functions operate on in-memory objects, which makes them suitable for
notebooks, batch jobs, or future services:

```python
from pathlib import Path

from src.predict import load_model, predict_image
from src.analyze import analyze_predictions
from src.spatial import extract_spatial_data
from src.heatmap import generate_density_grid

image_path = Path("data/raw/test_images/garden.jpg")
result = predict_image(load_model(), image_path)

summary = analyze_predictions(result)
spatial_data = extract_spatial_data(result)
height, width = result.orig_shape
density_grid = generate_density_grid(spatial_data, width, height, grid_size=8)
```

The `src.` imports work when the notebook or Python process starts from the
repository root. For notebooks launched elsewhere, add the repository root to
the Python path first.

## Outputs

Generated outputs are written to `results/`:

- Annotated prediction: `results/<input-image-filename>`
- Density heatmap: `results/density_heatmap.png`

Keep generated images and model weights out of commits unless there is a
deliberate artifact-versioning process for the project.

## Development notes

- Keep prediction, analysis, spatial, density, and plotting responsibilities
  separate. Extend the relevant module instead of duplicating its logic in a
  command-line script.
- Use `pathlib.Path` for project paths and keep reusable functions independent
  of command-line parsing and filesystem access where possible.
- Preserve the `bud` and `flower` class names used by the trained model and
  dataset configuration.
- Add tests or small reproducible scripts when changing data transformations,
  coordinate conventions, or density-grid behavior.
