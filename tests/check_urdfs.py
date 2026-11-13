"""
=============================================================================
  AI-Based Vision-Guided Robotic Grasping System: Deliverable 2
  Diagnostic Test: Object Catalog & Hardware Constraint Verification
  Lead: Satyam Kumar
=============================================================================
Usage:
    python tests/check_urdfs.py
=============================================================================
"""
import sys
import os
import numpy as np
import pybullet as p
import pybullet_data

# Ensure PickBot-sim root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulation.object_spawner import ObjectSpawner


def test_object_catalog():
    print("=" * 85)
    print("  RUNNING DIAGNOSTIC TEST: check_urdfs.py")
    print("  Verifying Franka Panda Gripper Compatibility & Bounding Dimensions")
    print("  Constraint: Graspable span MUST be between 25 mm and 60 mm (Gripper max = 80mm)")
    print("=" * 85)

    client_id = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    spawner = ObjectSpawner(table_height=0.0)
    catalog = spawner.catalog

    print(f"\nFound {len(catalog)} objects in curated catalog:\n")
    header = f"{'Name':<18} | {'Type':<18} | {'Dims (X x Y x Z) mm':<22} | {'Mass (kg)':<10} | {'Span (mm)':<10} | {'Status'}"
    print(header)
    print("-" * len(header))

    all_compliant = True

    for entry in catalog:
        key = entry["key"]
        name = entry["name"]
        otype = entry["type"]
        mass = entry["mass"]
        grasp_span_target = entry["grasp_span"] * 1000.0  # mm

        # Spawn individual test object
        body_id = None
        if otype == "primitive_box":
            body_id = spawner._create_primitive_box(entry["dims"], [1, 0, 0, 1], [0.5, 0.0, 0.05], [0, 0, 0, 1], mass)
        elif otype == "primitive_cylinder":
            body_id = spawner._create_primitive_cylinder(entry["dims"], [0, 1, 0, 1], [0.5, 0.0, 0.05], [0, 0, 0, 1], mass)
        elif otype == "compound_l_bracket":
            body_id = spawner._create_l_bracket([0, 0, 1, 1], [0.5, 0.0, 0.05], [0, 0, 0, 1], mass)
        elif otype == "urdf":
            body_id = spawner._load_urdf_object(entry["urdf_rel"], entry["scale"], [1, 1, 0, 1], [0.5, 0.0, 0.05], [0, 0, 0, 1], mass)

        if body_id is None:
            print(f"{name:<18} | {otype:<18} | {'LOAD FAILED':<22} | {mass:<10.2f} | {'N/A':<10} | [FAILED]")
            all_compliant = False
            continue

        # Measure AABB bounding box dimensions in mm
        aabb = p.getAABB(body_id)
        dims_m = np.array(aabb[1]) - np.array(aabb[0])
        dims_mm = np.round(dims_m * 1000.0, 1)
        dims_str = f"{dims_mm[0]:.0f} x {dims_mm[1]:.0f} x {dims_mm[2]:.0f}"

        # Graspable span verification (must be in [25mm, 60mm])
        span_ok = (25.0 <= grasp_span_target <= 60.0)
        mass_ok = (0.05 <= mass <= 0.20)
        compliant = span_ok and mass_ok

        if not compliant:
            all_compliant = False

        status_str = "[PASS] COMPLIANT" if compliant else "[FAIL] NON-COMPLIANT"
        print(f"{name:<18} | {otype:<18} | {dims_str:<22} | {mass:<10.2f} | {grasp_span_target:<10.1f} | {status_str}")

        p.removeBody(body_id)

    p.disconnect(client_id)

    print("-" * len(header))
    if all_compliant:
        print("\n>>> ALL OBJECTS STRICTLY CONFORM TO FRANKA PANDA HARDWARE CONSTRAINTS (25-60mm)! <<<\n")
    else:
        print("\n>>> WARNING: SOME OBJECTS EXCEED GRIPPER TOLERANCES! <<<\n")


if __name__ == "__main__":
    test_object_catalog()

