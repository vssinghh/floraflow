# FloraFlow: Minimalist Flow Matching Action Chunker for Tabletop Manipulation

`FloraFlow` is an end-to-end, cleanroom robotics system demonstrating 6-DoF object grasping, spatial transport, and fluid pouring using Optimal Transport Conditional Flow Matching (CFM) action chunking.

Built for the Franka Emika Panda robot arm in MuJoCo physics with zero external diffusion framework dependencies.

---

## 1. First-Principles Architecture

### What is it?
Conditional Flow Matching is a generative modeling framework that learns a continuous velocity vector field pushing a simple Gaussian noise distribution $p_0(x) = \mathcal{N}(0, I)$ directly onto an empirical trajectory distribution $p_1(x)$ along straight-line paths.

Instead of predicting single delta actions step by step, the policy predicts an entire future action chunk $X = [a_t, a_{t+1}, \dots, a_{t+H-1}] \in \mathbb{R}^{H \times D_{act}}$ ($H=16$, $D_{act}=8$) conditioned on current physical observations.

### Why do we use it?
Traditional behavioral cloning with Mean Squared Error suffers from the multimodality collapse problem: if an expert can reach around an obstacle from either the left or the right, MSE averages both trajectories, guiding the robot straight into the collision.

Standard Denoising Diffusion Probabilistic Models (DDPM) solve multimodality but require 50 to 100 iterative denoising steps along curved Brownian trajectories, imposing 100 to 500 ms of latency that violates real-time control constraints.

Optimal Transport Conditional Flow Matching defines straight-line probability paths:
$$x_t = (1 - (1 - \sigma_{min}) t) x_0 + t x_1$$
The target vector field velocity is constant along each sample path:
$$u_t(x_1 | x_0) = x_1 - (1 - \sigma_{min}) x_0$$
Because the paths are straight, numerical ODE integration converges in only 5 to 10 Euler steps, achieving sub-10 ms inference latencies well within the 50 ms deadline for 20 Hz control loops.

### How do we use it?
1. **Simulation & Synthesis**: Franka Panda arm manipulates a watering can on a table, transports it to a potted plant, and tilts the spout at 50 degrees to deposit water particles into the soil pot.
2. **Dataset Generation**: 100 deterministic expert trajectories ($17,400$ total transitions) collected via Damped Least Squares Inverse Kinematics and minimum-jerk interpolation, stored in structured HDF5.
3. **Flow Matching Training**: Lightweight residual MLP policy ($<1\text{M}$ parameters) trained on rolling 16-step action chunks using optimal transport vector field regression.
4. **Closed-Loop Evaluation**: Policy evaluated in MuJoCo across in-distribution and out-of-distribution initial configurations, logging success rates, kinematic clearances, and inference latencies.

### Which teams and real-world systems use it?
Flow matching and action chunking form the core execution heads of Google DeepMind (RoboCat, ALOHA 2), Meta FAIR, Physical Intelligence ($\pi_0$), and TRI (Diffusion Policy).

---

## 2. Directory Structure

```text
floraflow/
├── assets/
│   ├── franka_emika_panda/     # Franka arm & gripper MuJoCo models
│   └── scenes/
│       └── desk_scene.xml      # Tabletop scene: arm, plant, watering can, water particles
├── checkpoints/                # Model weights, configs, and dataset normalization stats
├── data/                       # HDF5 demonstration datasets
├── floraflow/
│   ├── env/
│   │   └── desk_env.py         # MuJoCo simulation environment (20 Hz control, 500 Hz physics)
│   ├── expert/
│   │   ├── ik_solver.py        # DLS 7-DoF Inverse Kinematics with nullspace regularization
│   │   ├── trajectory.py       # Minimum-jerk quintic polynomial interpolator
│   │   └── pour_planner.py     # Deterministic 9-phase pick, lift, transport, and pour planner
│   ├── policy/
│   │   ├── model.py            # FlowMatchingPolicy network with sinusoidal time embeddings
│   │   └── flow_matching.py    # Optimal Transport CFM vector field head & Euler integrator
│   ├── data/
│   │   └── dataset.py          # HDF5 rolling chunk dataset loader and normalizer
│   └── eval/
│       └── evaluator.py        # Closed-loop evaluation harness and benchmark scorecard
├── scripts/
│   ├── 01_generate_demos.py    # Generates 100 expert demonstrations to HDF5
│   ├── 02_train_policy.py      # Trains CFM action chunker policy
│   └── 03_evaluate_policy.py   # Closed-loop benchmark across ID and OOD scenarios
├── tests/                      # Pytest suite covering physics, math, and kinematics
└── pyproject.toml              # Dependencies and build configuration
```

---

## 3. Quick Start

### Installation
```bash
git clone https://github.com/vssinghh/floraflow.git
cd floraflow
uv venv .venv
source .venv/bin/activate
uv pip install -e .
```

### 1. Collect Demonstrations
Generate 100 expert demonstrations ($17,400$ state-action transitions):
```bash
uv run scripts/01_generate_demos.py --num-demos 100 --output data/watering_demos_100.h5
```

### 2. Train Flow Matching Policy
Train the 840K-parameter vector field policy head on Apple Silicon MPS or CUDA GPU:
```bash
uv run scripts/02_train_policy.py --epochs 100 --batch-size 256
```

### 3. Evaluate Closed-Loop Policy
Benchmark closed-loop execution across both in-distribution and out-of-distribution spatial configurations:
```bash
uv run scripts/03_evaluate_policy.py --num-episodes 20 --mode both
```

### 4. Run Test Suite
Run automated unit tests covering environment contracts, IK convergence, and flow matching calculus:
```bash
uv run pytest tests/
```

---

## 4. Benchmark Results & Scorecards

Closed-loop evaluation conducted across 20 held-out in-distribution trials and 20 out-of-distribution spatial perturbation trials (where can and plant positions were shifted outside training bounds).

<p align="center">
  <img src="assets/eval_policy_pour.png" width="48%" alt="Closed Loop Pour 3rd Person View"/>
  <img src="assets/eval_policy_pour_closeup.png" width="48%" alt="Closed Loop Pour Closeup View"/>
</p>

### Verification Scorecard

| Evaluation Metric | In-Distribution (Held-Out Seeds) | Out-of-Distribution (Spatial Perturbations) | Real-Time Requirement |
| :--- | :--- | :--- | :--- |
| **Total Evaluation Episodes** | 20 episodes | 20 episodes | - |
| **Task Success Rate** | **65.0%** (13 / 20) | **60.0%** (12 / 20) | > 50% |
| **Mean Maximum Tilt Angle** | **82.0°** | **88.2°** | > 40.0° |
| **Mean Spout Alignment Error** | 15.2 cm | 18.9 cm | < 20.0 cm |
| **Mean Inference Latency** | **3.67 ms** | **3.65 ms** | **< 50.0 ms (20 Hz)** |
| **Real-Time Control Constraint** | **PASS** | **PASS** | Sub-15 ms target |

### Technical Highlights
1. **Zero External Framework Dependencies**: The entire Flow Matching calculus (optimal transport probability paths, analytical velocity vector fields, and explicit Euler numerical ODE integration) is implemented directly in pure PyTorch.
2. **Sub-4ms Inference Latency**: At 3.67 ms per ODE solve (10 Euler integration steps), policy inference consumes less than 8% of the 50 ms control interval at 20 Hz, leaving ample headroom for virtual vision sensing.
3. **Action Chunk Execution Horizon**: Executing $K=12$ steps from each predicted $H=16$ chunk suppresses high-frequency feedback jitter, preserving smooth minimum-jerk kinematic execution.

