# AI-Based Vision-Guided Robotic Grasping System

> **Course:** Cyber Physical Systems — Semester 5
>
> **Topic:** AI-Based Vision-Guided Robotic Grasping System for Efficient Object Picking

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Mathematical Foundations](#mathematical-foundations)
- [Directory Structure](#directory-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Deliverable 1 Demonstration](#deliverable-1-demonstration)
  - [Interactive Playground](#interactive-playground)
  - [Benchmark Suite](#benchmark-suite)
  - [Fine-Tuning on Simulation Data](#fine-tuning-on-simulation-data)
- [Module Reference](#module-reference)
- [Object Catalog](#object-catalog)
- [Benchmark Results](#benchmark-results)
- [Deliverables](#deliverables)
- [Tech Stack](#tech-stack)

---

## Project Overview

- This project develops a **vision-based robotic picking system** that integrates AI-driven grasp detection with simulated robot execution
- A **Convolutional Neural Network (GG-CNN)** trained on the **Cornell Grasp Dataset** predicts optimal grasp points, gripper angles, and grasp quality scores from depth images
- A **PyBullet-simulated Franka Emika Panda robot arm** autonomously picks objects based on the CNN predictions
- The system is evaluated via **grasp success rate**, **inference latency**, and **Cornell validation metrics (IoU ≥ 0.25, Δθ ≤ 30°)**

---

## Key Features

- **Real-Time CNN Grasp Detection**
  - GG-CNN (Generative Grasping CNN) architecture with encoder-decoder topology
  - Produces pixel-wise heatmaps: grasp quality `Q(u,v)`, orientation angle `θ(u,v)`, and gripper width `W(u,v)`
  - GPU-accelerated inference at **~40 FPS** on NVIDIA RTX 3050 (~25ms latency)

- **Diverse Object Simulation**
  - 9+ named URDF objects: lego bricks, mugs, ducks, teddy bears, soccerballs, dominos, jenga blocks, cubes
  - 30 random mesh shapes from PyBullet's `random_urdfs` collection (1000 available)
  - Procedural primitives: boxes, cylinders, bars, spheres, capsules
  - Vibrant 10-color palette with collision-aware placement

- **3D World Deprojection**
  - Pinhole camera intrinsic matrix converts 2D pixel coordinates `(u, v)` + metric depth → 3D Cartesian coordinates `(X, Y, Z)` in robot frame
  - Calibrated overhead RGB-D camera (300×300, 45° FOV)

- **Full Pick-and-Place Execution**
  - 5-phase grasp sequence: approach → descend → close gripper → lift → transport to tray
  - Franka Panda 7-DOF inverse kinematics (IK) solver
  - Real-time grasp success tracking with statistics

- **Interactive Simulation Playground**
  - PyBullet GUI with interactive robot control sliders (X, Y, Z, Yaw)
  - Action buttons: Detect Grasps, Execute Grasp, Respawn Objects, Reset Robot
  - Live OpenCV perception dashboard (4-panel: RGB, depth, quality heatmap, angle map)
  - 3D grasp visualization with crosshairs, approach lines, and angle indicators

- **Benchmarking Suite**
  - Evaluated against 3 baselines: Random Grasp, Centroid Heuristic, Depth Antipodal Heuristic
  - Automated multi-trial evaluation across clutter levels (1, 3, 5 objects)
  - Publication-ready bar charts and comparison tables

- **Simulation Fine-Tuning**
  - Auto-labeled synthetic grasp dataset from PyBullet scenes
  - Fine-tuning loop with MSE multi-task loss (Q + cos + sin + width)
  - Converges in ~5 epochs (~6 seconds on RTX 3050)

---

## System Architecture

```
+-------------------------------------------------------------+
|                     PyBullet Simulation                     |
|  - Table workspace: X ∈ [0.38, 0.62], Y ∈ [-0.18, 0.18]    |
|  - Diverse objects: URDF meshes + procedural primitives      |
|  - Franka Emika Panda 7-DOF arm + parallel jaw gripper      |
|  - Tray for collecting grasped objects                       |
+-------------------------------------------------------------+
                              |
                     [Overhead Camera]
                   Eye: [0.5, 0.0, 0.85]m
                      FOV: 45°, 300×300px
                              |
             +----------------+----------------+
             |                                 |
       [RGB Stream]                      [Depth Field]
      (300 × 300 × 3)                 (Linear Metric Z)
             |                                 |
             +----------------+----------------+
                              |
                      [Depth Normalization]
              d_norm = clip(d − mean(d), −1, 1)
                              |
                    [GG-CNN Deep Network]
           Encoder: 3 Conv layers (32, 16, 8 filters)
           Decoder: 3 ConvTranspose (8, 16, 32 filters)
                              |
          +-------------------+-------------------+
          |                   |                   |
    [Quality Q(u,v)]    [Angle θ(u,v)]      [Width W(u,v)]
     Grasp confidence    Gripper orient.     Opening span
       heatmap [0,1]     [-π/2, π/2]        in pixels
          |                   |                   |
          +-------------------+-------------------+
                              |
             [Peak Detection + Local NMS Filtering]
                              |
             [Camera Deprojection: 2D → 3D World]
                              |
             [Franka Panda IK → Pick-and-Place]
```

---

## Mathematical Foundations

### Antipodal Grasp Representation

- Each grasp `g` in the image plane is parameterized as:
  - `g = (u, v, θ, w, q)`
- Where:
  - `(u, v)` — pixel center of the grasp in the camera image plane
  - `θ ∈ [-π/2, π/2]` — orientation angle of the gripper, computed via: `θ = ½ · atan2(sin 2θ, cos 2θ)`
  - `w` — gripper opening width in pixels
  - `q ∈ [0, 1]` — grasp quality score (predicted success probability)

### 2D-to-3D World Coordinate Deprojection

- Given camera intrinsic matrix `K`:
  ```
  K = | f_x   0   c_x |
      |  0   f_y  c_y |
      |  0    0    1  |
  ```
- Camera-frame coordinates:
  - `X_c = (u − c_x) · Z_c / f_x`
  - `Y_c = −(v − c_y) · Z_c / f_y`
- World-frame coordinates:
  - `X_w = X_cam + X_c`
  - `Y_w = Y_cam + Y_c`
  - `Z_w = Z_cam − Z_c`

### Cornell Validation Criteria

- A predicted grasp is considered a **valid match** against a ground-truth grasp if:
  - Grasp rectangle IoU ≥ 0.25
  - Orientation difference ≤ 30° (modulo 180°)

---

## Directory Structure

```
Cyber Physical Systems/
├── models/
│   ├── __init__.py
│   └── ggcnn.py                    # GG-CNN PyTorch architecture & multi-task loss
├── simulation/
│   ├── __init__.py
│   ├── camera.py                   # Calibrated overhead camera with 2D→3D deprojection
│   ├── object_spawner.py           # Diverse URDF + procedural object spawner with GT grasps
│   ├── robot.py                    # Franka Emika Panda 7-DOF arm loader & IK solver
│   └── environment.py             # PyBullet picking scene (table, tray, workspace markers)
├── perception/
│   ├── __init__.py
│   ├── grasp_geometry.py           # Grasp data class, Cornell IoU & angle metrics
│   ├── detector.py                 # Real-time grasp detection engine & NMS
│   └── visualizer.py              # 4-panel dashboard and OpenCV HUD generator
├── evaluation/
│   ├── __init__.py
│   ├── baselines.py                # Random, Centroid, and Depth heuristic baselines
│   └── benchmark.py               # Automated benchmarking suite & chart generator
├── training/
│   ├── __init__.py
│   ├── dataset.py                  # Auto-labeled PyBullet synthetic grasp dataset
│   └── train.py                   # Fine-tuning and training loop
├── checkpoints/
│   ├── ggcnn_weights_cornell/      # Cornell pre-trained weights (epoch 23)
│   │   └── ggcnn_epoch_23_cornell_statedict.pt
│   └── ggcnn_finetuned.pt          # Fine-tuned PyBullet checkpoint
├── outputs/
│   ├── deliverable1_clutter_1.png  # 4-panel demo: 1 object
│   ├── deliverable1_clutter_3.png  # 4-panel demo: 3 objects
│   ├── deliverable1_clutter_5.png  # 4-panel demo: 5 objects
│   └── benchmark_results.png      # Benchmark comparison bar charts
├── run_deliverable1.py             # Main Deliverable 1 entry point
├── playground.py                   # Interactive simulation playground
├── requirements.txt
└── README.md
```

---

## Prerequisites

- **OS:** Ubuntu 22.04+ / WSL2 with Ubuntu 24.04
- **Python:** 3.10+
- **GPU:** NVIDIA GPU with CUDA support (tested on RTX 3050, CUDA 12.4)
- **Display:** X11 / WSLg for GUI mode (PyBullet GUI + OpenCV windows)

---

## Installation

- Clone the repository:
  ```bash
  git clone <repository-url>
  cd "Cyber Physical Systems"
  ```

- Create and activate a virtual environment:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

- Verify GPU availability:
  ```bash
  python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
  ```

- Download pre-trained weights (if not included):
  - Cornell GG-CNN weights should be placed at: `checkpoints/ggcnn_weights_cornell/ggcnn_epoch_23_cornell_statedict.pt`

---

## Usage

### Deliverable 1 Demonstration

- **Run multi-clutter demonstration** (generates 4-panel dashboard snapshots for 1, 3, and 5 objects):
  ```bash
  python run_deliverable1.py --clutter-test --align-robot
  ```

- **Run with PyBullet 3D GUI window:**
  ```bash
  python run_deliverable1.py --gui --num-objects 3 --align-robot
  ```

- **Run continuous live perception loop** (real-time OpenCV dashboard):
  ```bash
  python run_deliverable1.py --gui --continuous --num-objects 4
  ```
  - Keyboard controls in the live loop:
    - `r` — respawn objects with a random count
    - `d` — save a snapshot of the current detection to `outputs/`
    - `q` — quit the live loop

- **All CLI flags:**
  | Flag | Description |
  |------|-------------|
  | `--gui` | Launch PyBullet 3D GUI window |
  | `--num-objects N` | Number of objects to spawn (1–8) |
  | `--clutter-test` | Run multi-scene sweep (1, 3, 5 objects) |
  | `--benchmark` | Run full baseline benchmarking suite |
  | `--align-robot` | Position Franka arm above detected grasp |
  | `--continuous` | Run real-time perception loop with live dashboard |
  | `--checkpoint PATH` | Path to model weights file |

### Interactive Playground

- **Launch the interactive playground:**
  ```bash
  python playground.py --num-objects 5
  ```

- **PyBullet GUI Controls (right-side panel):**
  - `Robot X / Y / Z` — move the robot arm end-effector in real time
  - `Gripper Yaw (deg)` — rotate the gripper orientation
  - `Num Objects (respawn)` — set the object count for the next respawn

- **Action Buttons:**
  - `DETECT GRASPS` — runs CNN inference on the current scene, draws grasp markers in 3D, updates OpenCV dashboard
  - `EXECUTE TOP GRASP` — performs a full 5-phase pick sequence (approach → descend → close gripper → lift → drop in tray)
  - `RESPAWN OBJECTS` — clears the scene and spawns new randomized objects
  - `RESET ROBOT` — returns the robot arm to its home position

- **OpenCV Dashboard Window:**
  - Displays a 4-panel view: RGB with grasp overlays, depth field, quality heatmap, angle map
  - Updates each time DETECT GRASPS is clicked

- **Playground CLI flags:**
  | Flag | Description |
  |------|-------------|
  | `--num-objects N` | Initial number of objects (1–8) |
  | `--no-dashboard` | Disable the OpenCV dashboard window |
  | `--checkpoint PATH` | Path to model weights file |

### Benchmark Suite

- **Run the full benchmarking evaluation:**
  ```bash
  python run_deliverable1.py --benchmark
  ```
- Evaluates the CNN against 3 baselines over 20 randomized trials at clutter levels 1, 3, and 5
- Outputs:
  - Console table with per-method statistics
  - Bar chart saved to `outputs/benchmark_results.png`

### Fine-Tuning on Simulation Data

- **Fine-tune GG-CNN on auto-labeled PyBullet scenes:**
  ```bash
  python -c "from training.train import train_model; train_model(epochs=5, batch_size=8)"
  ```
- Generates synthetic depth maps with auto-labeled ground-truth grasp annotations
- Converges in ~5 epochs (~6 seconds on RTX 3050)
- Saves fine-tuned weights to `checkpoints/ggcnn_finetuned.pt`

---

## Module Reference

### `models/ggcnn.py`
- **`GGCNN`** — fully convolutional encoder-decoder network
  - Encoder: 3 convolutional layers (32 → 16 → 8 filters)
  - Decoder: 3 transposed convolutional layers (8 → 16 → 32 filters)
  - 4 prediction heads: quality `Q`, `cos 2θ`, `sin 2θ`, width `W`
- **`load_trained_model()`** — factory function to load weights with automatic channel adaptation

### `simulation/environment.py`
- **`GraspingEnvironment`** — PyBullet scene manager
  - Loads ground plane, table, tray, and Franka Panda robot
  - Draws green workspace boundary markers in GUI mode
  - Provides `reset()`, `step()`, `get_observation()`, `get_ground_truth_grasps()`

### `simulation/object_spawner.py`
- **`ObjectSpawner`** — diverse object spawner
  - Builds a catalog of 9+ named URDF objects + 30 random meshes
  - 50/50 chance of URDF mesh vs procedural primitive per spawn
  - Collision-aware placement with minimum separation
  - Computes ground-truth grasp annotations for evaluation

### `simulation/robot.py`
- **`FrankaPandaRobot`** — 7-DOF Franka Emika Panda with parallel jaw gripper
  - `move_to_cartesian(pos, yaw)` — IK-based end-effector positioning
  - `reset()` — return to home posture
  - `get_ee_pose()` — current end-effector state

### `simulation/camera.py`
- **`OverheadCamera`** — simulated overhead RGB-D camera
  - Captures RGB (300×300×3), metric depth, and segmentation mask
  - `deproject_pixel_to_world(u, v, depth)` — 2D → 3D conversion
  - `project_world_to_pixel(xyz)` — 3D → 2D projection
  - Pre-computed intrinsic matrix `K`

### `perception/detector.py`
- **`GraspDetector`** — real-time CNN inference engine
  - Morrison-style depth normalization: `clip(d − mean(d), −1, 1)`
  - Gaussian-smoothed quality map with peak local max detection
  - Object masking and border suppression
  - Returns ranked `Grasp` objects + telemetry + heatmaps

### `perception/grasp_geometry.py`
- **`Grasp`** — antipodal grasp data class
  - Oriented rectangle computation and jaw line extraction
  - Rotated rectangle IoU via OpenCV
  - Cornell validation: `is_valid_match(gt, iou_thresh=0.25, angle_thresh=30°)`

### `perception/visualizer.py`
- **`Visualizer`** — 4-panel publication dashboard
  - Panel 1: RGB with grasp rectangle overlays and jaw contacts
  - Panel 2: Colorized metric depth field (Turbo colormap)
  - Panel 3: Grasp quality heatmap (Inferno colormap) with peak markers
  - Panel 4: Gripper orientation angle map (HSV colormap)
  - Top HUD banner with live telemetry

### `evaluation/baselines.py`
- **Random Grasp** — uniformly random pixel + angle selection
- **Centroid Heuristic** — object centroid detection via depth segmentation
- **Depth Antipodal Heuristic** — edge-gradient orientation + Sobel filtering

### `evaluation/benchmark.py`
- Automated multi-trial benchmarking runner
- Generates comparison bar charts (matplotlib)
- Outputs per-method: success rate, mean quality, latency, throughput

### `training/dataset.py`
- **`SyntheticGraspDataset`** — auto-labeled PyBullet scene generator
- Renders depth maps with ground-truth quality, angle, and width labels

### `training/train.py`
- **`train_model()`** — fine-tuning loop with MSE multi-task loss
- Saves best checkpoint to `checkpoints/ggcnn_finetuned.pt`

---

## Object Catalog

- The spawner uses a diverse mix of objects from **PyBullet's asset library**:

  | Object | Source | Description |
  |--------|--------|-------------|
  | Lego Brick | `lego/lego.urdf` | Classic interlocking brick |
  | Domino | `domino/domino.urdf` | Rectangular domino tile |
  | Jenga Block | `jenga/jenga.urdf` | Wooden-textured Jenga piece |
  | Mug | `objects/mug.urdf` | Coffee mug with handle |
  | Duck | `duck_vhacd.urdf` | Rubber duck mesh |
  | Teddy Bear | `teddy_vhacd.urdf` | Soft toy bear mesh |
  | Soccerball | `soccerball.urdf` | Spherical ball |
  | Cube | `cube_small.urdf` | Small rigid cube |
  | Block | `block.urdf` | Generic rigid block |
  | Random Meshes | `random_urdfs/` | 30 diverse random shapes from a pool of 1000 |
  | Box | Procedural | Randomized box primitive |
  | Cylinder | Procedural | Randomized cylinder primitive |
  | Bar | Procedural | Elongated rectangular bar |
  | Sphere | Procedural | Randomized sphere primitive |
  | Capsule | Procedural | Rounded cylinder primitive |

---

## Benchmark Results

- Evaluated over **20 randomized trials** across clutter levels (1, 3, and 5 objects):

  | Method | Success Rate (%) | Mean Quality Q | Latency (ms) | Throughput (FPS) |
  |--------|:---:|:---:|:---:|:---:|
  | Random Grasp | 41.7% | 0.296 | 0.26 | ~3800 |
  | Centroid Heuristic | 100.0% | 0.650 | 0.31 | ~3200 |
  | Depth Antipodal | 91.7% | 0.720 | 1.13 | ~880 |
  | **GG-CNN (Ours)** | **80.0%** | **0.530–0.865** | **2.47** | **~405** |

- **Key insights:**
  - At **2.47 ms** on RTX 3050, the CNN operates well within the 10 ms real-time threshold for 100 Hz closed-loop robotic control
  - Unlike heuristics that guess object centers, GG-CNN produces **dense, continuous probability heatmaps** over the entire scene
  - Pre-training on Cornell + Morrison-style depth normalization enables zero-shot generalization to simulated scenes

---

## Deliverables

### Deliverable 1 — Vision-Based Grasp Detection ✅

- Simulated robotic picking environment with diverse objects
- Trained GG-CNN model analyzes depth images and predicts:
  - Precise grasp points `(u, v)` → `(X, Y, Z)`
  - Gripper angles `θ`
  - Grasp quality scores `Q ∈ [0, 1]`
- Real-time inference at ~40 FPS on GPU
- Interactive playground with live visualization
- Benchmarked against 3 baselines

### Deliverable 2 — Autonomous Pick-and-Place Execution (Planned)

- Predicted 3D grasp coordinates used as robot arm waypoints
- Motion sequence:
  - Pre-grasp approach (10 cm above target)
  - Descend to grasp depth
  - Close gripper (force-controlled)
  - Lift to z = 0.3m
  - Transport to tray / drop bin
- Evaluated via:
  - Grasp success rate (object lifted and placed)
  - Contact point verification
  - Joint force feedback analysis

---

## Tech Stack

- **Deep Learning:** PyTorch 2.0+, CUDA 12.4
- **Physics Simulation:** PyBullet 3.2+
- **Computer Vision:** OpenCV 5.0+, scikit-image
- **Visualization:** Matplotlib, OpenCV HUD
- **Robot Model:** Franka Emika Panda (7-DOF, pybullet_data URDF)
- **Dataset:** Cornell Grasp Dataset (pre-trained), Synthetic PyBullet (fine-tuned)
- **Hardware:** NVIDIA RTX 3050 (GPU), WSL2 Ubuntu 24.04
