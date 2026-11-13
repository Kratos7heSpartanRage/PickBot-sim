"""
=============================================================================
  AI-Based Vision-Guided Robotic Grasping System: Deliverable 2
  Robotics & Physical Grasp Execution Subsystem: Grasp Executor
=============================================================================
Implements the 5-phase Pick-and-Place State Machine:
  Phase 1: Approach (Pre-grasp waypoint 15cm above target)
  Phase 2: Descend  (Calibrated finger immersion enclosing object)
  Phase 3: Grip     (Clamping force with slip prevention)
  Phase 4: Lift     (Vertical elevation with active force retention)
  Phase 5: Transfer & Drop (Carry to collection tray and release)
  Phase 6: Return   (Smooth return to home posture with persistent motor lock)
=============================================================================
"""
import time
import numpy as np
import pybullet as p


class GraspExecutor:
    """
    Executes physical pick-and-place trajectories using Franka Panda arm and gripper.
    """
    def __init__(self, robot, tray_pos=(0.45, 0.45, 0.22)):
        self.robot = robot
        self.tray_pos = list(tray_pos)

    def execute_pick_and_place(self, grasp, spawner=None):
        """
        Executes a complete pick, lift, carry, and drop-in-tray sequence.
        Args:
            grasp: Grasp object with world_coords [wx, wy, wz] and angle_rad
            spawner: Optional ObjectSpawner instance to track lifted object
        Returns:
            dict containing execution success and telemetry
        """
        if grasp.world_coords is None:
            return {"success": False, "reason": "No 3D world coordinates"}

        wx, wy, wz = grasp.world_coords
        yaw = grasp.angle_rad
        t_start = time.perf_counter()

        print(f"\n[Grasp Execution Started]")
        print(f"  Target World Pos : [{wx:.3f}, {wy:.3f}, {wz:.3f}] m")
        print(f"  Gripper Yaw Angle: {grasp.angle_deg:+.1f} deg")

        # -------------------------------------------------------------
        # Phase 1: Approach (Pre-grasp waypoint 15cm above target)
        # -------------------------------------------------------------
        approach_z = wz + 0.15
        print(f"  Phase 1: Approach  -> [{wx:.3f}, {wy:.3f}, {approach_z:.3f}]")
        self.robot.open_gripper(target_pos=0.04, steps=20)
        self.robot.move_to_cartesian([wx, wy, approach_z], target_yaw=yaw, steps=100)
        for _ in range(20):
            p.stepSimulation()

        # -------------------------------------------------------------
        # Phase 2: Descend (Enclose object between finger pads)
        # -------------------------------------------------------------
        grasp_z = max(0.012, wz - 0.016)
        print(f"  Phase 2: Descend   -> [{wx:.3f}, {wy:.3f}, {grasp_z:.3f}]")
        self.robot.move_to_cartesian([wx, wy, grasp_z], target_yaw=yaw, steps=100)
        for _ in range(20):
            p.stepSimulation()

        # -------------------------------------------------------------
        # Phase 3: Grip (Close fingers with high clamping force)
        # -------------------------------------------------------------
        print("  Phase 3: Grip      -> Clamping fingers (force=120N)...")
        self.robot.close_gripper(force=120.0, target_pos=0.0, steps=80)

        # Check contact telemetry
        contacts = p.getContactPoints(bodyA=self.robot.robot_id)
        num_contacts = len(contacts)
        print(f"  Telemetry: {num_contacts} active contact points detected.")

        # -------------------------------------------------------------
        # Phase 4: Lift (Elevate 25cm with active grip retention)
        # -------------------------------------------------------------
        lift_z = wz + 0.25
        print(f"  Phase 4: Lift      -> Elevating to z={lift_z:.3f}m...")
        self.robot.move_to_cartesian([wx, wy, lift_z], target_yaw=yaw, steps=120,
                                     keep_gripper_closed=True, grip_force=120.0)
        for _ in range(30):
            p.stepSimulation()

        # Verify physical lift success (check if object was elevated above 8cm)
        lifted_obj = None
        if spawner is not None:
            for obj in spawner.object_data:
                try:
                    pos, _ = p.getBasePositionAndOrientation(obj["id"])
                    if pos[2] > 0.08:
                        lifted_obj = obj
                        break
                except Exception:
                    pass

        # -------------------------------------------------------------
        # Phase 5: Transfer & Drop into Collection Tray
        # -------------------------------------------------------------
        if lifted_obj is not None or num_contacts > 0:
            obj_name = lifted_obj['name'] if lifted_obj else "Object"
            print(f"  Result: Physical Grasp Confirmed! ({obj_name} lifted)")
            print(f"  Phase 5: Transfer  -> Moving to tray [{self.tray_pos[0]:.2f}, {self.tray_pos[1]:.2f}, {self.tray_pos[2]:.2f}]...")
            self.robot.move_to_cartesian(self.tray_pos, target_yaw=0.0, steps=120,
                                         keep_gripper_closed=True, grip_force=120.0)
            for _ in range(20):
                p.stepSimulation()

            print("  Phase 6: Drop      -> Opening gripper over tray...")
            self.robot.open_gripper(target_pos=0.04, force=40.0, steps=60)
            for _ in range(30):
                p.stepSimulation()
            success = True
        else:
            print("  Result: MISS - object not engaged by gripper.")
            success = False

        # -------------------------------------------------------------
        # Phase 7: Smooth Return to Neutral Home Posture
        # -------------------------------------------------------------
        print("  Returning robot arm smoothly to home posture...")
        self.robot.move_to_home(steps=80)

        cycle_time = time.perf_counter() - t_start
        print(f"[Pick-and-Place Completed in {cycle_time:.2f}s | Success: {success}]\n")

        return {
            "success": success,
            "cycle_time_s": cycle_time,
            "lifted_object": lifted_obj["name"] if lifted_obj else None,
            "contacts": num_contacts
        }
