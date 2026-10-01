# FloraFlow

**FloraFlow** trains a 7-DoF Franka Emika Panda arm in MuJoCo to pick up a watering can, carry it across a desk, and tilt it to pour water into a plant pot. At `20 Hz` (`50 ms` per step), the policy takes three $128 \times 128$ RGB camera views (`third_person_cam`, `overhead_cam`, `wrist_cam`) and `8D` joint states, and runs a 10-step Flow Matching ODE solver in `5.2 ms` to predict a 16-step (`0.8 s`) action chunk.

<p align="center">
  <img src="assets/media/vla_telemetry_showcase.gif" width="96%" alt="FloraFlow 4-Layer Multi-Camera VLA Telemetry Rollout"/>
</p>

## 1. Telemetry & Camera Attention

[`VisionRolloutVisualizer`](floraflow/evaluation/visualizer.py) renders four internal policy signals at `20 Hz` across all three camera views:

<p align="center">
  <img src="assets/media/vla_telemetry_showcase_strip.png" width="98%" alt="6-Phase VLA Telemetry Contact Sheet"/>
</p>

1. **Spatial Keypoints (`+`)**: [`SpatialSoftmax`](floraflow/training/spatial_softmax.py) extracts 2D pixel coordinates from each camera's feature map. Trained end-to-end from action loss alone (no bounding-box labels), the highest-confidence points track the watering can handle, spout, plant leaves, and gripper fingers.
2. **Future Trajectory Ribbon (Cyan to Amber)**: MuJoCo forward kinematics (`mj_kinematics`) rolls out the predicted 16-step (`0.8 s`) joint chunk and projects the future 3D fingertip path onto each 2D camera frame.
3. **Camera Attention Weights**: [`MultiCameraCrossAttention`](floraflow/training/vision_model.py) shifts camera weights automatically by task phase:
   * **Approach (Step 1)**: `overhead_cam` carries `62.5%` of the attention weight to triangulate $(X, Y)$ positions on the desk.
   * **Grasp (Step 39)**: `wrist_cam` peaks (`41.3%` on ID, `51.8%` on Hard OOD) as the fingers close around the `8 mm` handle.
   * **Pour (Steps 116 to 196)**: When the tilted can blocks `wrist_cam`, weight shifts back to `third_person_cam` and `overhead_cam` (`85%` to `97%` combined) to hold the spout `5 to 8 cm` over the pot rim.
4. **Task Telemetry Strip**: Bottom plots track spout-to-pot distance (`cm`), can tilt angle (`deg`), gripper state (`OPEN` / `GRASPED`), and water particles inside the pot.

## 2. System Specifications

Predicting single motor steps causes compounding errors and jitter during transport and pouring, while standard MSE loss averages across valid trajectories. FloraFlow uses Optimal Transport Conditional Flow Matching (OT-CFM) to map Gaussian noise to 16-step action chunks along straight velocity paths in 10 Euler integration steps, and blends overlapping chunks across consecutive control ticks ($w_h = \exp(-0.05 h)$).

| Component | Specification |
| :- | :- |
| **Simulation & Control** | MuJoCo 3.x, 7-DoF Franka Panda arm + parallel gripper, `500 Hz` physics (`dt = 2 ms`), `20 Hz` policy control |
| **Inputs (Vision Policy)** | Three $128 \times 128$ RGB cameras (`third_person_cam`, `overhead_cam`, `wrist_cam`) + `8D` joint state (`7` joints + `1` gripper) |
| **Inputs (State Policy)** | Ground-truth 3D object coordinates + `8D` robot joint state |
| **Vision Backbone** | 4-layer ConvNet per camera + [`SpatialSoftmax`](floraflow/training/spatial_softmax.py) (`16` 2D keypoints, `32`-dim projection) + 4-head [`MultiCameraCrossAttention`](floraflow/training/vision_model.py) (`192`-dim fused vector) |
| **Policy Head & Solver** | Flow Matching ResMLP (`1.09M` params), 10-step Euler ODE solver (`~5.2 ms` on Apple Silicon MPS), sliding-window Temporal Ensembling |
| **Action Output** | 16-step horizon (`0.8 s`): absolute joints `joint_abs` (`8D`), relative deltas `joint_delta` (`8D`), or 6-DoF fingertip pose `eef_se3` (`10D`) |
| **Datasets & Checkpoints** | **Vision**: 300 clean demos + 300 domain-randomized demos · **State**: 500 demos · [**Google Drive Archive**](https://drive.google.com/drive/folders/1H5BHfbyAeGmpzyOtLy23bXUdGH2hlc66?usp=sharing) |

## 3. Benchmark Results

Policies are evaluated in closed-loop MuJoCo rollouts across **In-Distribution (ID)** spawns and **Hard Out-of-Distribution (OOD)** spawns (can and plant pot shifted `5 to 10 cm` outside training bounds).

<p align="center">
  <img src="assets/eval_policy_pour.png" width="48%" alt="Closed Loop Pour 3rd Person View"/>
  <img src="assets/eval_policy_pour_closeup.png" width="48%" alt="Closed Loop Pour Closeup View"/>
</p>

| Model Checkpoint | Input Modality | In-Distribution (20 Seeds) | Hard Out-of-Distribution (50 Seeds) | Mean Spout Error (ID / OOD) | Inference Latency (`< 50 ms` limit) |
| :- | :- | :- | :- | :- | :- |
| **Phase 1: State Policy** | Ground-truth 3D coordinates + joints | **100.0%** (`20/20`) | **96.0%** (`48/50`) | `8.0 cm` / `9.8 cm` | **3.56 ms** |
| **Run 5: Vision Baseline** | 3 cameras + joints + timer (`100` demos, no aug) | 75.0% (`15/20`) | 30.0% (`6/20`) | `16.3 cm` / `31.5 cm` | 4.65 ms |
| **Run 12: Vision Bottleneck** | 3 cameras + joints + timer (`300` demos, `192D` bottleneck) | 95.0% (`19/20`) | **100.0%** (`50/50` Clean) | `8.7 cm` / **8.5 cm** | 5.22 ms |
| **Run 15c: Clean Studio Champion** | 3 cameras + clock-free `8D` joints (`300` trimmed demos) | **100.0%** (`20/20`) | **96.0%** (`48/50` Clean) | **8.0 cm** / `9.9 cm` | **5.21 ms** |
| **Run 17c: Sim-to-Real Champion** | 3 cameras + clock-free `8D` joints (`300` Clean + `300` DR) | **85.0%** (Clean) / **75.0%** (DR) | **82.0%** (Clean) / **58.0%** (DR, `102/140` = **72.9%** total) | `11.9 cm` / `15.4 cm` | 5.77 ms |

### Ablation Summary (17 Runs)

Full experiment logs and failure analysis across all 17 runs are in [`docs/optimization_experiment_log.md`](docs/optimization_experiment_log.md):

1. **Dataset Scale & Shift Augmentation (`30%` to `80%` OOD)**: With 100 demos and no image augmentation (`Run 5`), the CNN memorized background pixel coordinates and failed on OOD object spawns (`30%`). Scaling to 300 demos with $\pm 4$ pixel random shifts (`Run 7`) raised Hard OOD success to `80%`.
2. **Bottleneck Compression (`72%` to `100%` OOD)**: Widening the 3-camera fusion vector to `320` dimensions (`Run 9`) reached `100%` ID accuracy but overfit on OOD spawns (`72%`). Compressing each camera to `16` keypoints and a `192D` fused bottleneck (`Run 12`) forced the encoder to pass only 2D spatial coordinates, reaching `100%` (`50/50`) on Clean Hard OOD.
3. **Clock-Free Proprioception & Dwell Trimming (`35%` to `96%` OOD)**: Early runs included an episode timer (`t / T`) in the state vector, causing the policy to tilt its wrist on a fixed schedule even after missed grasps. Removing the timer dropped success to `35%` (`Run 15a`) because stationary pauses between motion phases shared identical inputs. Trimming stationary frames during dataset loading (`Run 15c`) restored **100% ID** and **96% Hard OOD** success using only camera images and joint angles.
4. **Domain-Randomized Co-Training (`5.7%` to `72.9%` Sim-to-Real)**: The clean studio model (`Run 15c`) scored `5.7%` (`4/70`) under randomized lighting, table colors, $\pm 1.2\text{ cm}$ camera mount shifts, and $0.7\times\text{ to }2.0\times$ can mass. Co-training on 300 clean + 300 domain-randomized demos with `K=4` stratified flow sampling (`Run 17c`) reduced training time by `1.52x` (`33.3 min`) and achieved **72.9% (`102/140`)** across all four evaluation splits.

## 4. Quickstart

### 1. Install
```bash
git clone https://github.com/vssinghh/floraflow.git
cd floraflow
uv venv .venv
source .venv/bin/activate
uv pip install -e .
```

### 2. Evaluate Pretrained Checkpoints & Render Telemetry GIFs
```bash
# Evaluate the Clean Studio Champion (Run 15c) on In-Distribution and Hard OOD spawns
uv run python -m floraflow.evaluation \
  --checkpoint checkpoints/run15c_clock_free_trimmed8d/best_vision_policy.pt \
  --mode both --ood-difficulty hard --episodes 20

# Evaluate the Sim-to-Real Generalist (Run 17c) across all 4 benchmark splits (140 episodes)
uv run python -m floraflow.evaluation \
  --checkpoint checkpoints/run17c_sim2real_dr_600_k4/best_vision_policy.pt \
  --mode all

# Render the 4-layer VLA telemetry rollout GIF and 6-phase contact sheet
uv run python -m floraflow.evaluation.visualize \
  --checkpoint checkpoints/run15c_clock_free_trimmed8d/best_vision_policy.pt \
  --mode id --seed 102 --output assets/media/vla_telemetry_showcase.gif
```

### 3. Collect Demonstrations & Train From Scratch

```bash
# Collect 300 clean multi-camera demos and 300 domain-randomized demos
uv run python -m floraflow.collection --num-demos 300 --output datasets/watering_demos_vision_3cam_300.h5
uv run python -m floraflow.collection --num-demos 300 --start-seed 300 --domain-rand --output datasets/watering_demos_vision_3cam_dr_300.h5

# Train the Clean Studio Vision Policy (Run 15c configuration)
uv run python -m floraflow.training \
  --data datasets/watering_demos_vision_3cam_300.h5 \
  --save-dir checkpoints/run15c_clock_free_trimmed8d

# Train the Sim-to-Real Generalist Vision Policy (Run 17c configuration)
uv run python -m floraflow.training \
  --data datasets/watering_demos_vision_3cam_300.h5 datasets/watering_demos_vision_3cam_dr_300.h5 \
  --save-dir checkpoints/run17c_sim2real_dr_600_k4

# Optional: Collect, train, and evaluate the Phase 1 State Policy (ground-truth 3D coordinates)
uv run python -m floraflow.collection --state --num-demos 500 --output datasets/watering_demos_widened_500.h5 --widened-bounds
uv run python -m floraflow.training --state --data datasets/watering_demos_widened_500.h5 --epochs 80
uv run python -m floraflow.evaluation --state --mode both --ood-difficulty hard --episodes 50
```

### 4. Run Unit Tests
```bash
uv run pytest tests/
```

## 5. Repository Structure

```text
floraflow/
├── assets/
│   ├── franka_emika_panda/     # Franka arm and gripper MuJoCo XML/mesh assets
│   ├── media/                  # Telemetry rollout GIFs and 6-phase contact sheets
│   └── scenes/desk_scene.xml   # Tabletop scene: arm, plant pot, watering can, water spheres
├── datasets/                   # HDF5 demonstration files (.h5), benchmark splits, Drive links
├── checkpoints/                # Saved model weights (run15c, run17c) and training configs
├── docs/                       # 17-run optimization log (optimization_experiment_log.md)
├── floraflow/
│   ├── common/                 # MuJoCo 20 Hz environment (env.py) and 7-DoF IK (kinematics.py)
│   ├── collection/             # Minimum-jerk pour planner and HDF5 data collectors
│   ├── training/               # Spatial Softmax, Multi-Camera Cross-Attention, Flow Matching trainer
│   └── evaluation/             # Closed-loop evaluators and 4-layer VLA telemetry visualizer
└── tests/                      # Pytest suite for kinematics, env, vision models, and telemetry
```
