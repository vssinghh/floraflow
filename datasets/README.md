# FloraFlow Demonstration Datasets

All raw HDF5 demonstration archives (`.h5`) are hosted in our public Google Drive archive so they can be downloaded locally without exceeding GitHub's `100 MB` file size limit.

* **FloraFlow Google Drive Root**: [`floraflow/`](https://drive.google.com/drive/folders/1H5BHfbyAeGmpzyOtLy23bXUdGH2hlc66?usp=sharing)
* **Datasets Subfolder**: [`floraflow/datasets/`](https://drive.google.com/drive/folders/1guVQ41Lgj5fnXVw6fcNSwlFRxCedAMV3?usp=sharing)

## Available Datasets

| Filename | Modality | Episodes | Transitions | File Size | Direct Link |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `watering_demos_vision_3cam_300.h5` | Multi-Camera RGB + Proprio + 3D Aux Pose | 300 | 52,200 (`47,103` trimmed) | `1.3 GB` | [Download](https://drive.google.com/file/d/1NLjtoQK0Ag-qZcBh4YjE3VDb_tcozU4f/view?usp=sharing) |
| `watering_demos_500.h5` | Low-Dim State (`27D`) | 500 | 87,000 | `37 MB` | [Download](https://drive.google.com/file/d/1qDbWwWps385TLTRzEIb90NGluQ4iZJqp/view?usp=sharing) |
| `watering_demos_100.h5` | Low-Dim State (`27D`) | 100 | 17,400 | `7.5 MB` | [Download](https://drive.google.com/file/d/1KkrsqRQLr_NquqWptErAa_CaiLWXFVOG/view?usp=sharing) |

## Downloading Datasets Locally

Download any archive from the [Google Drive Datasets folder](https://drive.google.com/drive/folders/1guVQ41Lgj5fnXVw6fcNSwlFRxCedAMV3?usp=sharing) into `datasets/`, or use `gdown`:

```bash
# Download the 300-demo multi-camera vision dataset (1.3 GB)
uvx gdown 1NLjtoQK0Ag-qZcBh4YjE3VDb_tcozU4f -O datasets/watering_demos_vision_3cam_300.h5
```
