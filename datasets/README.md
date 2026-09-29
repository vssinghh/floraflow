# FloraFlow Demonstration Datasets

All raw HDF5 demonstration archives (`.h5`) are hosted in our public Google Drive archive so they can be downloaded locally without exceeding GitHub's `100 MB` file size limit.

* **FloraFlow Google Drive Root**: [`floraflow/`](https://drive.google.com/drive/folders/1H5BHfbyAeGmpzyOtLy23bXUdGH2hlc66?usp=sharing)
* **Datasets Subfolder**: [`floraflow/datasets/`](https://drive.google.com/drive/folders/1guVQ41Lgj5fnXVw6fcNSwlFRxCedAMV3?usp=sharing)

## Available Datasets

| Filename | Modality | Episodes | Transitions | File Size | Notes / Link |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `watering_demos_vision_3cam_300.h5` | Multi-Camera RGB + Proprio + 3D Aux Pose (Clean Studio, `seeds 0..299`) | 300 | 52,200 (`47,103` trimmed) | `1.30 GB` | [Download](https://drive.google.com/file/d/1NLjtoQK0Ag-qZcBh4YjE3VDb_tcozU4f/view?usp=sharing) |
| `watering_demos_vision_3cam_dr_300.h5` | Multi-Camera RGB + Proprio + 3D Aux Pose (Sim-to-Real Domain Randomized, `seeds 300..605`) | 300 | 52,200 (`47,205` trimmed) | `1.57 GB` | Local / Google Drive Archive |
| `eval_sim2real_benchmark.json` | 4-Split Sim-to-Real Evaluation Manifest (`clean_id`, `clean_ood`, `dr_id`, `dr_ood`) | 140 eval configs | - | `219 KB` | Tracked directly in Git (`datasets/eval_sim2real_benchmark.json`) |
| `watering_demos_500.h5` | Phase 1 Historical Low-Dim State (`27D`, Widened) | 500 | 87,000 | `37.3 MB` | [Download](https://drive.google.com/file/d/1qDbWwWps385TLTRzEIb90NGluQ4iZJqp/view?usp=sharing) |
| `watering_demos_100.h5` | Phase 1 Historical Low-Dim State (`27D`, Narrow) | 100 | 17,400 | `7.5 MB` | [Download](https://drive.google.com/file/d/1KkrsqRQLr_NquqWptErAa_CaiLWXFVOG/view?usp=sharing) |

## Generating or Downloading Datasets Locally

Download any archive from the [Google Drive Datasets folder](https://drive.google.com/drive/folders/1guVQ41Lgj5fnXVw6fcNSwlFRxCedAMV3?usp=sharing) into `datasets/`, or generate them deterministically:

```bash
# Download the 300-demo clean studio multi-camera vision dataset (1.30 GB)
uvx gdown 1NLjtoQK0Ag-qZcBh4YjE3VDb_tcozU4f -O datasets/watering_demos_vision_3cam_300.h5

# Or generate the 300-demo Sim-to-Real Domain-Randomized dataset locally (1.57 GB)
uv run python -m floraflow.collection --num-demos 300 --start-seed 300 --domain-rand --output datasets/watering_demos_vision_3cam_dr_300.h5
```

