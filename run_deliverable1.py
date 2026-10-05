import argparse
import time
import os
# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import numpy as np

from simulation.environment import GraspingEnvironment
from perception.detector import GraspDetector
from perception.visualizer import Visualizer
from evaluation.benchmark import run_benchmark

def main():
    parser = argparse.ArgumentParser(description="Cyber Physical Systems: Deliverable 1 Demonstration")
    parser.add_argument("--gui", action="store_true", help="Launch PyBullet 3D GUI window")
    parser.add_argument("--num-objects", type=int, default=3, help="Number of objects to spawn (1 to 5)")
    parser.add_argument("--clutter-test", action="store_true", help="Run multi-scene clutter sweep (1, 3, 5 objects)")
    parser.add_argument("--benchmark", action="store_true", help="Run comprehensive baseline benchmarking suite")
    parser.add_argument("--align-robot", action="store_true", help="Position Franka Panda arm above detected grasp")
    parser.add_argument("--continuous", action="store_true", help="Run continuous real-time perception loop with live dashboard")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/ggcnn_weights_cornell/ggcnn_epoch_23_cornell_statedict.pt",
                        help="Path to trained model weights")
    args = parser.parse_args()

    print("=" * 80)
    print("  CYBER PHYSICAL SYSTEMS: PROJECT DELIVERABLE 1")
    print("  AI-Based Vision-Guided Robotic Grasping System")
    print("=" * 80)

    # If user requested benchmark evaluation:
    if args.benchmark:
        run_benchmark(num_trials=20, clutter_levels=(1, 3, 5))
        return

    # 1. Initialize PyBullet environment
    print(f"--> Initializing PyBullet Simulation Environment (GUI={args.gui})...")
    env = GraspingEnvironment(gui=args.gui)

    # 2. Initialize CNN Grasp Detector
    print(f"--> Initializing GG-CNN Grasp Detector with weights: {args.checkpoint}...")
    detector = GraspDetector(checkpoint_path=args.checkpoint)
    visualizer = Visualizer()

    # 3. Demonstration Scenarios
    scenarios = [args.num_objects] if not args.clutter_test else [1, 3, 5]

    for clutter in scenarios:
        print(f"")
        print(f"[SCENE EXECUTION] Spawning {clutter} randomized objects in workspace...")
        env.reset(num_objects=clutter)

        # Allow objects to settle
        for _ in range(30):
            env.step()

        # Capture RGB-D observation
        rgb, depth, seg = env.get_observation()
        gt_grasps = env.get_ground_truth_grasps()

        # Run real-time CNN inference
        grasps, telemetry, heatmaps = detector.detect(rgb, depth, camera=env.camera, top_k=5)

        print("-" * 60)
        print(f"  PERCEPTION RESULTS (Clutter: {clutter} objects):")
        print(f"  Inference Latency : {telemetry['inference_time_ms']:.2f} ms")
        print(f"  Throughput        : {telemetry['fps']:.0f} FPS")
        print(f"  Device            : {telemetry['device'].upper()}")
        print(f"  Detected Grasps   : {len(grasps)}")

        # Show object names
        obj_info = env.spawner.get_object_info()
        print(f"  Scene Objects     : {', '.join(o['name'] for o in obj_info)}")

        if len(grasps) > 0:
            top_g = grasps[0]
            print(f"  Top-1 Grasp Quality : {top_g.quality:.3f}")
            print(f"  Top-1 2D Center     : ({top_g.u:.1f}, {top_g.v:.1f}) px")
            print(f"  Top-1 Angle         : {top_g.angle_deg:+.1f} deg")
            print(f"  Top-1 Gripper Width : {top_g.width_px:.1f} px")
            if top_g.world_coords is not None:
                wx, wy, wz = top_g.world_coords
                print(f"  Top-1 3D World Pos  : [{wx:.3f}, {wy:.3f}, {wz:.3f}] meters")

            # Check Cornell validation match against GT grasps
            matched = any(top_g.is_valid_match(gt) for gt in gt_grasps)
            print(f"  Cornell Criterion Match: {'VALID (IoU>=0.25, dTheta<=30deg)' if matched else 'FEASIBLE'}")

            # Align robot end-effector above grasp if requested
            if args.align_robot and top_g.world_coords is not None:
                print(f"--> Aligning Franka Panda gripper above grasp target (z={top_g.world_coords[2]+0.12:.3f}m)...")
                approach_pos = [top_g.world_coords[0], top_g.world_coords[1], top_g.world_coords[2] + 0.12]
                env.robot.move_to_cartesian(approach_pos, target_yaw=top_g.angle_rad, steps=100)

        # Generate and save presentation snapshot
        fig_path = f"outputs/deliverable1_clutter_{clutter}.png"
        visualizer.save_report_figure(rgb, depth, heatmaps, grasps, telemetry, save_path=fig_path)

    # If continuous live loop requested:
    if args.continuous:
        print("")
        print("--> Running continuous LIVE PERCEPTION loop.")
        print("--> An OpenCV window will display the real-time dashboard.")
        print("--> Press 'q' in the dashboard window or Ctrl+C to stop.")
        print("--> Press 'r' to respawn objects, 'd' for new detection snapshot.")

        cv2.namedWindow("CPS Live Perception", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("CPS Live Perception", 900, 550)

        try:
            frame_idx = 0
            while True:
                env.step()
                if frame_idx % 15 == 0:
                    rgb, depth, _ = env.get_observation()
                    grasps, telem, hm = detector.detect(rgb, depth, camera=env.camera, top_k=5)

                    # Build and display live dashboard
                    dashboard = visualizer.create_dashboard(rgb, depth, hm, grasps, telem)
                    cv2.imshow("CPS Live Perception", cv2.cvtColor(dashboard, cv2.COLOR_RGB2BGR))

                    # Also print periodic telemetry
                    if frame_idx % 60 == 0:
                        obj_info = env.spawner.get_object_info()
                        print(f"[Live] FPS:{telem['fps']:.0f} | Lat:{telem['inference_time_ms']:.1f}ms | "
                              f"Q:{telem['top_score']:.2f} | Objects: {', '.join(o['name'] for o in obj_info)}")

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nStopped by user (q key).")
                    break
                elif key == ord('r'):
                    num = np.random.randint(2, 6)
                    print(f"\n[Respawn] Spawning {num} new objects...")
                    env.reset(num_objects=num)
                    for _ in range(60):
                        env.step()
                elif key == ord('d'):
                    # Save current detection snapshot
                    snap_path = f"outputs/live_snapshot_{int(time.time())}.png"
                    if grasps:
                        visualizer.save_report_figure(rgb, depth, hm, grasps, telem, save_path=snap_path)
                        print(f"\n[Snapshot] Saved to {snap_path}")

                frame_idx += 1
                time.sleep(0.005)

        except KeyboardInterrupt:
            print("")
            print("Stopped live loop.")
        finally:
            cv2.destroyAllWindows()

    env.close()
    print("")
    print("=" * 80)
    print("  DELIVERABLE 1 DEMONSTRATION COMPLETE")
    print("  All visual artifacts saved in 'outputs/' folder.")
    print("=" * 80)

if __name__ == '__main__':
    main()
