"""
=============================================================================
  CYBER PHYSICAL SYSTEMS: DELIVERABLE 2 PHYSICAL EXECUTION BENCHMARK
  Comprehensive Robotic Pick-and-Place Evaluation Against Baselines
=============================================================================
Compares physical execution performance across 4 manipulation methods:
  1. Random Grasp: Uniform stochastic candidate sampling
  2. Centroid Heuristic: Bounding contour center with moment-based orientation
  3. Depth Antipodal: Edge-pair gradient heuristic
  4. Our CNN (GG-CNN): Fully convolutional antipodal grasp synthesis

Evaluates:
  - Physical Pick & Place Success Rate (%) [lifted > 8cm and delivered to tray]
  - Force Closure Achievement Rate (%) [verified Coulomb friction cone margin]
  - Mean Friction Margin (%) [distance to Coulomb slip boundary]
  - End-to-End Cycle Time (seconds)
=============================================================================
"""
import time
import os
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulation.environment import GraspingEnvironment
from simulation.grasp_executor import GraspExecutor
from perception.detector import GraspDetector
from .baselines import RandomGraspDetector, CentroidGraspDetector, DepthAntipodalGraspDetector


def run_physical_benchmark(num_trials=10, clutter_levels=(1, 3, 5),
                           save_path="outputs/deliverable2_benchmark.png"):
    """
    Executes physical pick-and-place trials in PyBullet across all 4 methods
    and saves the Deliverable 2 comparison figure.
    """
    print("\n" + "=" * 78)
    print("  CYBER PHYSICAL SYSTEMS: DELIVERABLE 2 PHYSICAL EXECUTION BENCHMARK")
    print(f"  Trials per level: {num_trials} | Clutter sweeps: {clutter_levels}")
    print("=" * 78 + "\n")

    env = GraspingEnvironment(gui=False)
    executor = GraspExecutor(env.robot)

    cnn_detector = GraspDetector()
    random_det = RandomGraspDetector()
    centroid_det = CentroidGraspDetector()
    depth_det = DepthAntipodalGraspDetector()

    methods = [
        ("Random Grasp", random_det),
        ("Centroid Heuristic", centroid_det),
        ("Depth Antipodal", depth_det),
        ("GG-CNN (Ours)", cnn_detector)
    ]

    # Metrics dictionary
    stats = {
        m[0]: {
            "success": [],
            "force_closure": [],
            "margin_pct": [],
            "cycle_time": []
        }
        for m in methods
    }

    try:
        for clutter in clutter_levels:
            print(f"--> [Clutter Sweep: {clutter} Object(s)] Running {num_trials} trials per method...")

            for trial_idx in range(num_trials):
                # For fair comparison across methods, each trial uses the exact same randomized scene seed
                scene_seed = 42 + trial_idx * 17 + clutter * 1000
                for m_name, detector in methods:
                    # Reset scene deterministically per trial so all methods face the identical setup
                    random.seed(scene_seed)
                    np.random.seed(scene_seed)
                    env.reset(num_objects=clutter)
                    for _ in range(25):
                        env.step()

                    rgb, depth, seg = env.get_observation()
                    t_plan_start = time.perf_counter()

                    # Grasp prediction
                    if m_name == "GG-CNN (Ours)":
                        grasps, _, _ = detector.detect(rgb, depth, camera=env.camera, top_k=1)
                    else:
                        grasps, _ = detector.detect(rgb, depth, seg, camera=env.camera)

                    if not grasps or grasps[0].world_coords is None:
                        # Failed to find feasible grasp
                        stats[m_name]["success"].append(0.0)
                        stats[m_name]["force_closure"].append(0.0)
                        stats[m_name]["margin_pct"].append(0.0)
                        stats[m_name]["cycle_time"].append(time.perf_counter() - t_plan_start)
                        continue

                    target_grasp = grasps[0]

                    # Physical Pick-and-Place Execution
                    res = executor.execute_pick_and_place(target_grasp, spawner=env.spawner, save_plot=False)

                    is_succ = 1.0 if res.get("success") else 0.0
                    is_fc = 1.0 if res.get("force_closure") else 0.0
                    margin = float(res.get("friction_margin_pct", 0.0))
                    total_time = (time.perf_counter() - t_plan_start)

                    stats[m_name]["success"].append(is_succ)
                    stats[m_name]["force_closure"].append(is_fc)
                    stats[m_name]["margin_pct"].append(margin)
                    stats[m_name]["cycle_time"].append(total_time)

    finally:
        env.close()

    # -------------------------------------------------------------
    # Generate Publication-Quality Comparison Chart
    # -------------------------------------------------------------
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.2))
    fig.patch.set_facecolor("#0e1117")

    method_names = [m[0] for m in methods]
    colors = ["#6b7280", "#3b82f6", "#06b6d4", "#10b981"]

    # 1. Physical Pick & Place Success Rate (%)
    succ_rates = [float(np.mean(stats[n]["success"])) * 100.0 for n in method_names]
    ax1 = axes[0]
    ax1.set_facecolor("#161b22")
    bars1 = ax1.bar(method_names, succ_rates, color=colors, edgecolor="#30363d", alpha=0.9)
    ax1.set_title("Physical Pick Success Rate (%)", color="#f0f6fc", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Lift & Deliver Success (%)", color="#f0f6fc", fontsize=10)
    ax1.set_ylim(0, 115)
    for b in bars1:
        yval = b.get_height()
        ax1.text(b.get_x() + b.get_width() / 2.0, yval + 2, f"{yval:.1f}%",
                 ha="center", color="#f0f6fc", fontweight="bold", fontsize=9)

    # 2. Force Closure Achievement Rate (%)
    fc_rates = [float(np.mean(stats[n]["force_closure"])) * 100.0 for n in method_names]
    ax2 = axes[1]
    ax2.set_facecolor("#161b22")
    bars2 = ax2.bar(method_names, fc_rates, color=colors, edgecolor="#30363d", alpha=0.9)
    ax2.set_title("Force Closure Rate (%)", color="#f0f6fc", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Friction Cone Compliance (%)", color="#f0f6fc", fontsize=10)
    ax2.set_ylim(0, 115)
    for b in bars2:
        yval = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2.0, yval + 2, f"{yval:.1f}%",
                 ha="center", color="#f0f6fc", fontweight="bold", fontsize=9)

    # 3. Mean Friction Cone Margin (%)
    margins = [float(np.mean(stats[n]["margin_pct"])) for n in method_names]
    ax3 = axes[2]
    ax3.set_facecolor("#161b22")
    bars3 = ax3.bar(method_names, margins, color=colors, edgecolor="#30363d", alpha=0.9)
    ax3.set_title("Friction Cone Margin (%)", color="#f0f6fc", fontsize=11, fontweight="bold")
    ax3.set_ylabel(r"Safety Margin $(1 - \tan\theta/\mu)$ %", color="#f0f6fc", fontsize=10)
    ax3.set_ylim(0, 115)
    for b in bars3:
        yval = b.get_height()
        ax3.text(b.get_x() + b.get_width() / 2.0, yval + 2, f"{yval:.1f}%",
                 ha="center", color="#f0f6fc", fontweight="bold", fontsize=9)

    # 4. Total Cycle Time (seconds)
    cycle_times = [float(np.mean(stats[n]["cycle_time"])) for n in method_names]
    ax4 = axes[3]
    ax4.set_facecolor("#161b22")
    bars4 = ax4.bar(method_names, cycle_times, color=colors, edgecolor="#30363d", alpha=0.9)
    ax4.set_title("End-to-End Cycle Time (s)", color="#f0f6fc", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Perception + Motion (seconds)", color="#f0f6fc", fontsize=10)
    ax4.set_ylim(0, max(cycle_times) * 1.35 if cycle_times else 5.0)
    for b in bars4:
        yval = b.get_height()
        ax4.text(b.get_x() + b.get_width() / 2.0, yval + 0.1, f"{yval:.2f}s",
                 ha="center", color="#f0f6fc", fontweight="bold", fontsize=9)

    for ax in axes:
        ax.set_xticks(range(len(method_names)))
        ax.set_xticklabels(method_names, rotation=15, ha="right", color="#c9d1d9", fontsize=9)
        ax.tick_params(colors="#8b949e")
        ax.grid(axis="y", linestyle=":", alpha=0.3, color="#8b949e")

    plt.suptitle("Cyber Physical Systems: Deliverable 2 Physical Pick-and-Place Benchmark",
                 fontsize=14, fontweight="bold", color="#f0f6fc", y=1.00)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()

    # -------------------------------------------------------------
    # Print Executive Summary Table
    # -------------------------------------------------------------
    print("\n" + "=" * 84)
    print("  DELIVERABLE 2 PHYSICAL BENCHMARK SUMMARY TABLE")
    print("=" * 84)
    print(f"{'Method':<22} | {'Success Rate (%)':<17} | {'Force Closure (%)':<18} | {'Margin (%)':<11} | {'Cycle Time':<10}")
    print("-" * 84)
    for n, sr, fc, mg, ct in zip(method_names, succ_rates, fc_rates, margins, cycle_times):
        print(f"{n:<22} | {sr:15.1f}% | {fc:16.1f}% | {mg:9.1f}% | {ct:7.2f}s")
    print("=" * 84)
    print(f"\n[Deliverable 2 Benchmark Complete] Figure saved to: {save_path}\n")

    return stats
