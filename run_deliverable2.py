import argparse
import time
import os
import numpy as np
import cv2

from simulation.environment import GraspingEnvironment
from perception.detector import GraspDetector
from perception.visualizer import Visualizer
from simulation.grasp_executor import GraspExecutor

def run_single_pick_trial(env, detector, executor, visualizer=None, place_in_tray=True, verbose=True):
    """
    Executes a single autonomous vision-guided pick-and-place cycle.
    Returns:
        dict: Performance telemetry including detection metrics, execution success, contact forces, and cycle time.
    """
    t_start = time.perf_counter()
    
    # 1. Capture camera observation
    rgb, depth, seg = env.get_observation()
    
    # 2. Vision perception: GG-CNN grasp detection
    grasps, det_telemetry, heatmaps = detector.detect(rgb, depth, camera=env.camera, top_k=5)
    t_det = time.perf_counter()
    
    if len(grasps) == 0:
        if verbose:
            print("  [Perception] No feasible grasp candidates detected.")
        return {
            "success": False,
            "object": None,
            "inference_time_ms": det_telemetry.get("inference_time_ms", 0.0),
            "planning_execution_time_s": 0.0,
            "total_cycle_time_s": time.perf_counter() - t_start,
            "grasps_found": 0,
            "grip_force": 0.0,
            "lift_force": 0.0
        }
    
    top_grasp = grasps[0]
    if verbose:
        print(f"  [Perception] Top-1 Candidate: Q={top_grasp.quality:.3f}, "
              f"px=({top_grasp.u:.1f}, {top_grasp.v:.1f}), angle={top_grasp.angle_deg:+.1f} deg")
        if top_grasp.world_coords is not None:
            wx, wy, wz = top_grasp.world_coords
            print(f"               3D Target World: [{wx:.3f}, {wy:.3f}, {wz:.3f}] m")
            
    # 3. Actuation: Franka Panda pick-and-place execution
    t_exec_start = time.perf_counter()
    if hasattr(executor, "execute_pick_and_place"):
        result = executor.execute_pick_and_place(top_grasp, spawner=env.spawner)
    else:
        result = executor.execute(top_grasp, place_in_tray=place_in_tray, go_home=True)
    t_exec_end = time.perf_counter()
    
    obj_name = result.get("lifted_object") or result.get("object")
    telemetry = {
        "success": result.get("success", False),
        "object": obj_name,
        "inference_time_ms": det_telemetry.get("inference_time_ms", 0.0),
        "planning_execution_time_s": t_exec_end - t_exec_start,
        "total_cycle_time_s": t_exec_end - t_start,
        "grasps_found": len(grasps),
        "top_quality": top_grasp.quality,
        "force_closure": result.get("force_closure", False),
        "normal_force": result.get("normal_force", 0.0),
        "friction_force": result.get("friction_force", 0.0),
        "friction_angle_deg": result.get("friction_angle_deg", 0.0),
        "friction_margin_pct": result.get("friction_margin_pct", 0.0),
        "joint_torques": result.get("joint_torques", []),
        "peak_joint": result.get("peak_joint", "J4"),
        "peak_torque": result.get("peak_torque", 0.0),
        "profile_plot": result.get("profile_plot")
    }
    return telemetry


def main():
    parser = argparse.ArgumentParser(description="Cyber Physical Systems: Deliverable 2 Demonstration")
    parser.add_argument("--gui", action="store_true", help="Launch PyBullet 3D GUI window")
    parser.add_argument("--num-objects", type=int, default=3, help="Number of objects per scene (1 to 5)")
    parser.add_argument("--trials", type=int, default=5, help="Number of repeated autonomous pick trials")
    parser.add_argument("--clear-scene", action="store_true", help="Iteratively pick objects until table is cleared")
    parser.add_argument("--clutter-sweep", action="store_true", help="Evaluate across clutter levels 1, 3, and 5 objects")
    parser.add_argument("--no-tray", action="store_true", help="Skip placing objects in the collection tray")
    parser.add_argument("--benchmark", action="store_true", help="Run Deliverable 2 Physical Pick-and-Place Execution Benchmark against baselines")
    parser.add_argument("--benchmark-trials", type=int, default=10, help="Number of trials per clutter level for benchmark")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/ggcnn_weights_cornell/ggcnn_epoch_23_cornell_statedict.pt",
                        help="Path to trained GG-CNN model weights")
    args = parser.parse_args()

    # If user requested physical execution benchmark:
    if args.benchmark:
        from evaluation.physical_benchmark import run_physical_benchmark
        clutters = (1, 3, 5) if args.clutter_sweep else (args.num_objects,)
        run_physical_benchmark(num_trials=args.benchmark_trials, clutter_levels=clutters)
        return

    print("=" * 80)
    print("  CYBER PHYSICAL SYSTEMS: PROJECT DELIVERABLE 2")
    print("  Autonomous Closed-Loop Vision-Guided Robotic Pick-and-Place")
    print("=" * 80)

    # 1. Initialize simulation environment
    print(f"--> Initializing PyBullet Simulation Environment (GUI={args.gui})...")
    env = GraspingEnvironment(gui=args.gui)

    # 2. Initialize perception & execution
    print(f"--> Loading GG-CNN Grasp Detector ({args.checkpoint})...")
    detector = GraspDetector(checkpoint_path=args.checkpoint)
    executor = GraspExecutor(env.robot)
    visualizer = Visualizer()
    os.makedirs("outputs", exist_ok=True)

    clutter_levels = [1, 3, 5] if args.clutter_sweep else [args.num_objects]
    place_tray = not args.no_tray

    all_trials_results = []

    try:
        for clutter in clutter_levels:
            print("\n" + "#" * 70)
            print(f"  SCENARIO EVALUATION: Clutter Level = {clutter} Object(s)")
            print("#" * 70)

            clutter_successes = 0
            clutter_trials = 0

            for trial_idx in range(args.trials):
                print(f"\n--- [Trial {trial_idx + 1}/{args.trials}] Spawning {clutter} objects ---")
                env.reset(num_objects=clutter)
                for _ in range(60):
                    env.step()

                obj_info = env.spawner.get_object_info()
                print(f"  Spawned Objects: {', '.join(o['name'] for o in obj_info)}")

                if args.clear_scene:
                    print("  --> Mode: Iterative Scene Clearing...")
                    remaining = len(env.spawner.spawned_ids)
                    attempts = 0
                    max_attempts = remaining + 2

                    while remaining > 0 and attempts < max_attempts:
                        attempts += 1
                        print(f"\n  [Pick Attempt {attempts}] Objects remaining: {remaining}")
                        res = run_single_pick_trial(env, detector, executor, visualizer, place_tray, verbose=True)
                        clutter_trials += 1
                        if res["success"]:
                            clutter_successes += 1
                            remaining -= 1
                            print(f"  --> Successfully picked {res['object']}! (Remaining: {remaining})")
                        else:
                            print(f"  --> Attempt {attempts} failed to lift object.")
                else:
                    res = run_single_pick_trial(env, detector, executor, visualizer, place_tray, verbose=True)
                    clutter_trials += 1
                    if res["success"]:
                        clutter_successes += 1
                    all_trials_results.append(res)

                    print(f"  Result : {'SUCCESS' if res['success'] else 'FAILED'}")
                    print(f"  Object : {res['object']}")
                    print(f"  Cycle  : {res['total_cycle_time_s']:.2f} s "
                          f"(Inference: {res['inference_time_ms']:.1f} ms, Motion: {res['planning_execution_time_s']:.2f} s)")

            sr = (clutter_successes / max(1, clutter_trials)) * 100.0
            print("-" * 70)
            print(f"  Summary for Clutter Level {clutter}:")
            print(f"    Grasp Success Rate : {clutter_successes}/{clutter_trials} ({sr:.1f}%)")
            print("-" * 70)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        env.close()

    print("\n" + "=" * 80)
    print("  DELIVERABLE 2 DEMONSTRATION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()

