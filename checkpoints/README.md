# FloraFlow Model Checkpoints

All experimental model checkpoints (`.pt` weights and `.json` normalization statistics across Runs `0` through `16b`) are hosted in our public Google Drive archive, while the **Unified Champion Checkpoint (`run15c_clock_free_trimmed8d`)** is also tracked directly in this GitHub repository for immediate out-of-the-box evaluation and visualization.

* **FloraFlow Google Drive Root**: [`floraflow/`](https://drive.google.com/drive/folders/1H5BHfbyAeGmpzyOtLy23bXUdGH2hlc66?usp=sharing)
* **Checkpoints Subfolder**: [`floraflow/checkpoints/`](https://drive.google.com/drive/folders/1wVQT9tAf5RZB1_yUMw79dE4RrbnmoPjS?usp=sharing)

## Tracked in GitHub (Ready Out of the Box)

| Checkpoint Directory | Architecture & Action Space | ID Success (20 Seeds) | Hard OOD Success (50 Clean Seeds) | Size |
| :--- | :--- | :--- | :--- | :--- |
| [`run15c_clock_free_trimmed8d/`](run15c_clock_free_trimmed8d/) | Tri-Camera + 4-Head Cross-Attention, Clock-Free `8D` Proprio, Trimmed Grasp Dwell (`joint_abs`) | **100.0%** (`20 / 20`) | **96.0%** (`48 / 50`) | `13 MB` |

### Evaluate or Visualize the Tracked Champion Model
```bash
# Run closed-loop evaluation (ID + Hard OOD)
uv run python -m floraflow.evaluation --checkpoint checkpoints/run15c_clock_free_trimmed8d/best_vision_policy.pt --mode both --ood-difficulty hard --episodes 20

# Render 4-layer VLA telemetry GIF and 6-phase contact sheet
uv run python -m floraflow.evaluation.visualize --checkpoint checkpoints/run15c_clock_free_trimmed8d/best_vision_policy.pt --mode id --seed 102 --output assets/media/vla_telemetry_showcase.gif
```

## Full Experimental Archive on Google Drive

All 18 experimental runs (`run0_baseline` through `run16b_eef_se3_40ep`, totaling `253 MB`) are publicly accessible in [`floraflow/checkpoints/`](https://drive.google.com/drive/folders/1wVQT9tAf5RZB1_yUMw79dE4RrbnmoPjS?usp=sharing), including:
* `run12_bottleneck_3cam` (OOD Champion with synthetic clock: `100.0%` on Clean Hard OOD)
* `run14_clean_data_3cam` (Clean collision-validated dataset baseline: `100.0%` ID, `90.0%` Hard OOD)
* `run15c_clock_free_trimmed8d` (Unified clock-free `8D` champion: `100.0%` ID, `96.0%` Hard OOD)
* `run16a_joint_delta_trial60` (Relative `joint_delta` action chunks: `100.0%` ID, `88.0%` Hard OOD)
* `run16b_eef_se3_40ep` (Hardware-agnostic `10D` fingertip `SE(3)` + `rot6d` + 3D Pose Ruler Quiz: `80.0%` ID, `88.0%` Hard OOD)
