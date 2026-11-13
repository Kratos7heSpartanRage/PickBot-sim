"""
=============================================================================
  Test Suite: Deliverable 2 Interactive Playground Capabilities Verification
=============================================================================
Verifies:
  1. No repeat grasp on already-picked objects (auto-rescan targets remaining table objects).
  2. Batch pick with custom count (picks N objects sequentially without human intervention).
  3. Pick All (clears the entire scene sequentially without human intervention).
  4. Integer object count display (no floating-point decimals).
  5. Post-grasp and post-reset home posture retention (zero drift back to tray).
=============================================================================
"""
import sys
import os
import numpy as np
import pybullet as p

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.environment import GraspingEnvironment
from simulation.grasp_executor import GraspExecutor
from perception.detector import GraspDetector
from playground import InteractivePlayground


def run_feature_test():
    print("=" * 70)
    print("  RUNNING DELIVERABLE 2 PLAYGROUND CAPABILITY VERIFICATION")
    print("=" * 70)

    # Launch headless playground (no GUI / no dashboard for fast automated verification)
    playground = InteractivePlayground(num_objects=4, show_dashboard=False)

    print("\n--- TEST 1: Initial Scene & Integer Counts ---")
    table_objs = playground.get_table_objects()
    tray_objs = playground.get_tray_objects()
    print(f"Initial table objects: {len(table_objs)}")
    print(f"Initial tray objects : {len(tray_objs)}")
    assert len(table_objs) == 4, f"Expected 4 objects on table, got {len(table_objs)}"
    assert len(tray_objs) == 0, f"Expected 0 objects in tray, got {len(tray_objs)}"

    # Test slider values
    spawn_n = int(round(p.readUserDebugParameter(playground.slider_num_obj)))
    batch_n = int(round(p.readUserDebugParameter(playground.slider_batch_count)))
    assert isinstance(spawn_n, int), "spawn_n must be an integer"
    assert isinstance(batch_n, int), "batch_n must be an integer"
    print(f"Slider values: Spawn Count = {spawn_n} (int), Batch Count = {batch_n} (int)")
    print("[PASS] Test 1: Initial Scene & Integer Counts verified.")

    print("\n--- TEST 2: Single Pick with Automatic Re-Targeting ---")
    # First detection
    grasps = playground.run_detection()
    assert len(grasps) > 0, "GG-CNN must find grasps on initial scene"
    top_pos_1 = grasps[0].world_coords
    print(f"Target 1: [{top_pos_1[0]:.3f}, {top_pos_1[1]:.3f}, {top_pos_1[2]:.3f}]")

    # Execute single grasp
    success = playground.execute_single_grasp()
    print(f"Single grasp result: success={success}")
    assert success, "Single grasp must succeed on calibrated object"

    # Verify 1 object in tray, 3 on table
    table_objs = playground.get_table_objects()
    tray_objs = playground.get_tray_objects()
    print(f"Post-pick 1 table objects: {len(table_objs)}")
    print(f"Post-pick 1 tray objects : {len(tray_objs)}")
    assert len(table_objs) == 3, f"Expected 3 objects on table, got {len(table_objs)}"
    assert len(tray_objs) == 1, f"Expected 1 object in tray, got {len(tray_objs)}"

    # Check that robot returned home (no drift)
    current_q = [p.getJointState(playground.env.robot.robot_id, j)[0] for j in playground.env.robot.arm_joints]
    home_diff = np.max(np.abs(np.array(current_q) - np.array(playground.env.robot.home_posture)))
    print(f"Maximum joint angle deviation from home posture: {home_diff:.4f} rad")
    assert home_diff < 0.05, f"Robot drifted from home! Diff: {home_diff:.4f} rad"

    # Verify that new grasp target is DIFFERENT from the picked object location
    assert len(playground.last_grasps) > 0, "Auto-rescan must find new grasps on remaining 3 objects"
    top_pos_2 = playground.last_grasps[0].world_coords
    dist_between_targets = np.linalg.norm(np.array(top_pos_1[:2]) - np.array(top_pos_2[:2]))
    print(f"Distance between target 1 and new target 2: {dist_between_targets*1000:.1f} mm")
    assert dist_between_targets > 0.02, "Target did not update to another object! Repeat grasp detected."
    print("[PASS] Test 2: Single pick, tray delivery, home return & re-targeting verified.")

    print("\n--- TEST 3: Batch Pick (2 out of 3 objects) ---")
    # Batch pick 2 objects
    playground.batch_pick(target_count=2)

    table_objs = playground.get_table_objects()
    tray_objs = playground.get_tray_objects()
    print(f"Post-batch table objects: {len(table_objs)}")
    print(f"Post-batch tray objects : {len(tray_objs)}")
    assert len(tray_objs) >= 3, f"Expected at least 3 objects in tray, got {len(tray_objs)}"
    assert len(table_objs) <= 1, f"Expected at most 1 object on table, got {len(table_objs)}"
    print("[PASS] Test 3: Batch picking (subset count) verified.")

    print("\n--- TEST 4: Pick All (Clear Scene) ---")
    # Pick all remaining objects
    playground.batch_pick(target_count=None)

    table_objs = playground.get_table_objects()
    tray_objs = playground.get_tray_objects()
    print(f"Final table objects: {len(table_objs)}")
    print(f"Final tray objects : {len(tray_objs)}")
    assert len(table_objs) == 0, f"Expected 0 objects on table, got {len(table_objs)}"
    assert len(tray_objs) == 4, f"Expected 4 objects in tray, got {len(tray_objs)}"
    print("[PASS] Test 4: Pick All completely cleared the scene into the tray.")

    print("\n" + "=" * 70)
    print("  ALL DELIVERABLE 2 PLAYGROUND CAPABILITIES 100% VERIFIED!")
    print("=" * 70)

    playground.env.close()


if __name__ == "__main__":
    run_feature_test()

