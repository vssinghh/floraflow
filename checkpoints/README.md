# FloraFlow Model Checkpoints

All experimental model checkpoints (`.pt` weights and `.json` normalization statistics across Runs `0` through `16b`) are hosted in our public Google Drive archive, while the **Unified Champion Checkpoint (`run15c_clock_free_trimmed8d`)** is also tracked directly in this GitHub repository for immediate out-of-the-box evaluation and visualization.

* **FloraFlow Google Drive Root**: [`floraflow/`](https://drive.google.com/drive/folders/1H5BHfbyAeGmpzyOtLy23bXUdGH2hlc66?usp=sharing)
* **Checkpoints Subfolder**: [`floraflow/checkpoints/`](https://drive.google.com/drive/folders/1wVQT9tAf5RZB1_yUMw79dE4RrbnmoPjS?usp=sharing)

## Tracked in GitHub (Ready Out of the Box)

| Checkpoint Directory | Architecture & Action Space | ID Success (20 Seeds) | Hard OOD Success (50 Clean Seeds) | Size |
| :--- | :--- | :--- | :--- | :--- |
| [`run15c_clock_free_trimmed8d/`](run15c_clock_free_trimmed8d/) | Clean Studio Specialist: Tri-Camera + 4-Head Cross-Attention, Clock-Free `8D` Proprio (`joint_abs`, `16` kp) | **100.0%** (`20 / 20`) | **96.0%** (`48 / 50`) | `13 MB` |
| [`run17c_sim2real_dr_600_k4/`](run17c_sim2real_dr_600_k4/) | Sim-to-Real Generalist Champion: 600 Demos (`300 Clean + 300 DR`), `K=4` Stratified Flow, `32` kp, Locked `VisionTrainConfig` | **85.0%** Clean / **75.0%** DR-ID | **82.0%** Clean / **58.0%** DR-OOD (`102/140` = `72.9%`) | `13 MB` |

### Train, Evaluate, or Visualize Checkpoints
```bash
# Train 600-episode Sim-to-Real Generalist using locked VisionTrainConfig defaults
uv run python -m floraflow.training \
  --data datasets/watering_demos_vision_3cam_300.h5 datasets/watering_demos_vision_3cam_dr_300.h5 \
  --save-dir checkpoints/run17c_sim2real_dr_600_k4

# Run closed-loop evaluation (Clean ID + Hard OOD)
uv run python -m floraflow.evaluation --checkpoint checkpoints/run15c_clock_free_trimmed8d/best_vision_policy.pt --mode both --ood-difficulty hard --episodes 20

# Run 4-split Sim-to-Real evaluation (clean_id, clean_ood, dr_id, dr_ood)
uv run python -m floraflow.evaluation --checkpoint checkpoints/run17c_sim2real_dr_600_k4/best_vision_policy.pt --mode all

# Render 4-layer VLA telemetry GIF and 6-phase contact sheet
uv run python -m floraflow.evaluation.visualize --checkpoint checkpoints/run15c_clock_free_trimmed8d/best_vision_policy.pt --mode id --seed 102 --output assets/media/vla_telemetry_showcase.gif
```

## Full Experimental Archive on Google Drive

All experimental runs (`run0_baseline` through `run17c_sim2real_dr_600_k4`, totaling `279 MB`) are accessible in [`floraflow/checkpoints/`](https://drive.google.com/drive/folders/1wVQT9tAf5RZB1_yUMw79dE4RrbnmoPjS?usp=sharing), including:
* `run12_bottleneck_3cam` (OOD Champion with synthetic clock: `100.0%` on Clean Hard OOD)
* `run14_clean_data_3cam` (Clean collision-validated dataset baseline: `100.0%` ID, `90.0%` Hard OOD)
* `run15c_clock_free_trimmed8d` (Unified clock-free `8D` Clean Studio specialist: `100.0%` ID, `96.0%` Hard OOD)
* `run16a_joint_delta_trial60` (Relative `joint_delta` action chunks: `100.0%` ID, `88.0%` Hard OOD)
* `run16b_eef_se3_40ep` (Hardware-agnostic `10D` fingertip `SE(3)` + `rot6d` + 3D Pose Ruler Quiz: `80.0%` ID, `88.0%` Hard OOD)
* `run17_sim2real_dr_600` (Sim-to-Real Domain-Randomized 600-demo baseline: `85.0%` Clean ID, `76.0%` Clean OOD, `80.0%` DR-ID, `54.0%` DR-OOD, `98/140` = `70.0%` overall)
* `run17c_sim2real_dr_600_k4` (Sim-to-Real Generalist Champion with `K=4` Stratified Multi-Sample Flow Amortization, shortcut-free `192D` bottleneck, and locked `VisionTrainConfig`: `85.0%` Clean ID, `82.0%` Clean OOD, `75.0%` DR-ID, `58.0%` DR-OOD, `102/140` = `72.9%` overall in `33.3 min`)

