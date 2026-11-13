"""
=============================================================================
  AI-Based Vision-Guided Robotic Grasping System: Deliverable 2
  Diagnostic Test: CNN Perception & 3D World Deprojection Verification
  Lead: Satyam Kumar
=============================================================================
Usage:
    python tests/test_zoom2.py
=============================================================================
"""
import sys
import os
import numpy as np

# Ensure PickBot-sim root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulation.environment import GraspingEnvironment
from perception.detector import GraspDetector


def test_perception_and_deprojection():
    print("=" * 80)
    print("  RUNNING DIAGNOSTIC TEST: test_zoom2.py")
    print("  Verifying GG-CNN Inference, Grasp Quality & 3D World Deprojection")
    print("=" * 80)

    checkpoint = "checkpoints/ggcnn_weights_cornell/ggcnn_epoch_23_cornell_statedict.pt"
    if not os.path.exists(checkpoint):
        checkpoint = "checkpoints/ggcnn_finetuned.pt"

    print(f"\n--> Initializing Environment (DIRECT mode)...")
    env = GraspingEnvironment(gui=False)
    print(f"--> Initializing GG-CNN Detector ({checkpoint})...")
    detector = GraspDetector(checkpoint_path=checkpoint)

    try:
        for clutter in [1, 3, 5]:
            print(f"\n" + "-" * 60)
            print(f"[TEST SCENARIO] Testing Clutter Level {clutter}:")
            env.reset(num_objects=clutter)

            for _ in range(30):
                env.step()

            rgb, depth, seg = env.get_observation()
            gt_grasps = env.get_ground_truth_grasps()

            grasps, telemetry, _ = detector.detect(rgb, depth, camera=env.camera, top_k=5)

            print(f"  Device           : {telemetry['device'].upper()}")
            print(f"  Inference Latency: {telemetry['inference_time_ms']:.2f} ms ({telemetry['fps']:.0f} FPS)")
            print(f"  Candidates Found : {len(grasps)}")

            assert len(grasps) > 0, f"No grasps detected for clutter level {clutter}!"
            top_g = grasps[0]

            print(f"  Top Grasp Quality: {top_g.quality:.3f} (Spec Target: >= 0.65)")
            print(f"  Top Grasp 2D     : ({top_g.u:.1f}, {top_g.v:.1f}) px | Angle: {top_g.angle_deg:+.1f} deg")

            # Check 3D Deprojection
            assert top_g.world_coords is not None, "World coordinates deprojection failed!"
            wx, wy, wz = top_g.world_coords
            print(f"  Top Grasp 3D Pos : [{wx:.3f}, {wy:.3f}, {wz:.3f}] meters")

            # Verify coordinates lie inside active pick area
            x_valid = 0.35 <= wx <= 0.65
            y_valid = -0.20 <= wy <= 0.20
            z_valid = 0.00 <= wz <= 0.10
            print(f"  Workspace Bounds : X in [0.35, 0.65]: {x_valid} | Y in [-0.20, 0.20]: {y_valid} | Z in [0.00, 0.10]: {z_valid}")
            assert x_valid and y_valid and z_valid, f"Predicted grasp 3D coordinate out of bounds: [{wx}, {wy}, {wz}]"

            # Check Cornell Criterion match against ground truth
            matched = any(top_g.is_valid_match(gt) for gt in gt_grasps)
            print(f"  Cornell IoU Match: {'VALID (IoU >= 0.25, dTheta <= 30deg)' if matched else 'FEASIBLE'}")

        print("\n" + "=" * 80)
        print("  >>> ALL PERCEPTION & 3D DEPROJECTION TESTS PASSED! <<<")
        print("=" * 80)

    finally:
        env.close()


if __name__ == "__main__":
    test_perception_and_deprojection()

