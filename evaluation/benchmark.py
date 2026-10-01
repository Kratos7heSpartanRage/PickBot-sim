import numpy as np
import matplotlib.pyplot as plt
import os
import time

from simulation.environment import GraspingEnvironment
from perception.detector import GraspDetector
from .baselines import RandomGraspDetector, CentroidGraspDetector, DepthAntipodalGraspDetector

def run_benchmark(num_trials=20, clutter_levels=(1, 3, 5), save_path="outputs/benchmark_results.png"):
    print("=== Starting Grasp Detection Benchmark ===")
    print(f"Trials per level: {num_trials}, Clutter levels: {clutter_levels}")

    env = GraspingEnvironment(gui=False)
    cnn_detector = GraspDetector()
    random_det = RandomGraspDetector()
    centroid_det = CentroidGraspDetector()
    depth_det = DepthAntipodalGraspDetector()

    methods = [
        ('Random Grasp', random_det),
        ('Centroid Heuristic', centroid_det),
        ('Depth Heuristic', depth_det),
        ('Our CNN (GG-CNN)', cnn_detector)
    ]

    results = {m[0]: {'quality': [], 'latency': [], 'valid': []} for m in methods}

    for clutter in clutter_levels:
        print(f"--> Testing clutter level: {clutter} objects...")
        for trial in range(num_trials):
            env.reset(num_objects=clutter)
            rgb, depth, seg = env.get_observation()
            gt_grasps = env.get_ground_truth_grasps()
            obj_ids = set(env.spawner.spawned_ids)

            for name, detector in methods:
                if name == 'Our CNN (GG-CNN)':
                    grasps, telem, _ = detector.detect(rgb, depth, camera=env.camera, top_k=1)
                else:
                    grasps, telem = detector.detect(rgb, depth, seg, camera=env.camera)

                latency = telem.get('inference_time_ms', 1.0)
                quality = telem.get('top_score', 0.0)

                # Check if detected grasp is geometrically valid (placed on actual object)
                is_valid = False
                if len(grasps) > 0:
                    pred_g = grasps[0]
                    u_idx = int(round(np.clip(pred_g.u, 0, seg.shape[1] - 1)))
                    v_idx = int(round(np.clip(pred_g.v, 0, seg.shape[0] - 1)))
                    
                    # On object segmentation or close in 3D Euclidean space
                    on_seg = seg[v_idx, u_idx] in obj_ids
                    close_to_gt = False
                    for gt in gt_grasps:
                        if pred_g.world_coords is not None and gt.world_coords is not None:
                            dist = np.linalg.norm(pred_g.world_coords[:2] - gt.world_coords[:2])
                            if dist < 0.05:
                                close_to_gt = True
                                break

                    if (on_seg or close_to_gt) and quality >= 0.30:
                        is_valid = True

                results[name]['latency'].append(latency)
                results[name]['quality'].append(quality)
                results[name]['valid'].append(1.0 if is_valid else 0.0)

    env.close()

    # Generate benchmark visualization
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    names = [m[0] for m in methods]
    colors = ['#888888', '#4A90E2', '#50E3C2', '#2ECC71']

    # 1. Grasp Success / Validity Rate
    success_rates = [float(np.mean(results[n]['valid'])) * 100.0 for n in names]
    axes[0].set_xticks(range(len(names)))
    bars1 = axes[0].bar(names, success_rates, color=colors, edgecolor='black', alpha=0.85)
    axes[0].set_title("Grasp Success / Validity Rate (%)", fontsize=13, fontweight='bold')
    axes[0].set_ylabel("Success Rate (%)", fontsize=11)
    axes[0].set_ylim(0, 108)
    for bar in bars1:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval:.1f}%", ha='center', fontweight='bold')

    # 2. Average Grasp Quality Score
    avg_qualities = [float(np.mean(results[n]['quality'])) for n in names]
    axes[1].set_xticks(range(len(names)))
    bars2 = axes[1].bar(names, avg_qualities, color=colors, edgecolor='black', alpha=0.85)
    axes[1].set_title("Mean Grasp Quality Score Q", fontsize=13, fontweight='bold')
    axes[1].set_ylabel("Quality Score (0 to 1)", fontsize=11)
    axes[1].set_ylim(0, 1.08)
    for bar in bars2:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.2f}", ha='center', fontweight='bold')

    # 3. Inference Time (Latency ms)
    avg_latencies = [float(np.mean(results[n]['latency'])) for n in names]
    axes[2].set_xticks(range(len(names)))
    bars3 = axes[2].bar(names, avg_latencies, color=colors, edgecolor='black', alpha=0.85)
    axes[2].set_title("Inference / Planning Latency (ms)", fontsize=13, fontweight='bold')
    axes[2].set_ylabel("Latency (milliseconds)", fontsize=11)
    for bar in bars3:
        yval = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f"{yval:.2f} ms", ha='center', fontweight='bold')

    for ax in axes:
        ax.set_xticklabels(names, rotation=15, ha='right', fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    plt.suptitle("Cyber Physical Systems: Grasp Detection Benchmark Comparison", fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"Benchmark completed successfully! Figure saved to: {save_path}")
    print("")
    print("Summary Statistics Table:")
    print("-" * 75)
    print(f"{'Method':<25} | {'Success Rate (%)':<18} | {'Mean Quality Q':<15} | {'Latency (ms)':<12}")
    print("-" * 75)
    for n in names:
        sr = float(np.mean(results[n]['valid'])) * 100.0
        mq = float(np.mean(results[n]['quality']))
        lat = float(np.mean(results[n]['latency']))
        print(f"{n:<25} | {sr:>16.1f}% | {mq:>15.3f} | {lat:>10.2f} ms")
    print("-" * 75)

    return results
