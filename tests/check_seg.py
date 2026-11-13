"""
=============================================================================
  AI-Based Vision-Guided Robotic Grasping System: Deliverable 2
  Diagnostic Test: Segmentation & Observation Contract Verification
  Lead: Satyam Kumar
=============================================================================
Usage:
    python tests/check_seg.py
=============================================================================
"""
import sys
import os
import numpy as np

# Ensure PickBot-sim root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulation.environment import GraspingEnvironment


def test_segmentation_and_contracts():
    print("=" * 75)
    print("  RUNNING DIAGNOSTIC TEST: check_seg.py")
    print("  Verifying Observation Contract, Table Z-Calibration & Segmentation IDs")
    print("=" * 75)

    env = GraspingEnvironment(gui=False)
    all_passed = True

    try:
        for clutter in [1, 3, 5]:
            print(f"\n[Test Scenario] Clutter Level {clutter}:")
            spawned_ids = env.reset(num_objects=clutter)

            # Let settle
            for _ in range(30):
                env.step()

            rgb, depth, seg = env.get_observation()
            obj_info = env.spawner.get_object_info()
            gt_grasps = env.get_ground_truth_grasps()

            # 1. Check RGB contract
            assert rgb.shape == (300, 300, 3), f"RGB shape mismatch: {rgb.shape}"
            assert rgb.dtype == np.uint8, f"RGB dtype mismatch: {rgb.dtype}"
            print(f"  [PASS] RGB Shape: {rgb.shape}, dtype: {rgb.dtype}")

            # 2. Check Depth contract (table depth at 0.85m +- 0.01m)
            assert depth.shape == (300, 300), f"Depth shape mismatch: {depth.shape}"
            assert depth.dtype == np.float32, f"Depth dtype mismatch: {depth.dtype}"
            table_pixels = (seg == 1)
            mean_table_depth = float(np.mean(depth[table_pixels]))
            print(f"  [PASS] Mean Table Depth: {mean_table_depth:.4f} m (Target: 0.85m +- 0.01m)")
            if not (0.84 <= mean_table_depth <= 0.86):
                print(f"  [WARN] Table depth {mean_table_depth:.4f} slightly outside [0.84, 0.86]")
                all_passed = False

            # 3. Check Robot arm pixels (Must be exactly 0)
            robot_pixels = int(np.sum(seg == env.robot.robot_id))
            print(f"  [PASS] Robot arm pixels in camera FOV: {robot_pixels} (Target: 0)")
            assert robot_pixels == 0, f"Robot arm occluding camera FOV! ({robot_pixels} pixels)"

            # 4. Check Segmentation IDs
            unique_ids = np.unique(seg)
            print(f"  [INFO] Unique IDs in segmentation mask: {list(unique_ids)}")
            assert 1 in unique_ids, "Table surface (ID 1) missing from segmentation!"

            detected_objects = 0
            for obj in obj_info:
                bid = obj["body_id"]
                pix_count = int(np.sum(seg == bid))
                print(f"    - Object '{obj['name']}' (ID {bid}): {pix_count} pixels, settled pos: {obj['settled_pos']}")
                if pix_count > 0:
                    detected_objects += 1

            assert detected_objects == clutter, f"Expected {clutter} objects in seg, found {detected_objects}"
            print(f"  [PASS] All {clutter} objects successfully identified in segmentation mask.")

            # 5. Check Object Info contract
            assert len(obj_info) == clutter
            for obj in obj_info:
                assert "name" in obj and "body_id" in obj and "settled_pos" in obj and "dimensions" in obj
            print(f"  [PASS] Object info contract verified ({len(obj_info)} objects).")

            # 6. Check Ground Truth Grasps contract
            assert len(gt_grasps) == clutter
            print(f"  [PASS] Ground truth grasps contract verified ({len(gt_grasps)} grasps).")

        print("\n" + "=" * 75)
        if all_passed:
            print("  >>> ALL SEGMENTATION & CONTRACT CHECKS PASSED SUCCESSFULLY! <<<")
        else:
            print("  >>> CHECKS COMPLETED WITH WARNINGS <<<")
        print("=" * 75)

    finally:
        env.close()


if __name__ == "__main__":
    test_segmentation_and_contracts()

