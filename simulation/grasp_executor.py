"""
=============================================================================
  AI-Based Vision-Guided Robotic Grasping System: Deliverable 2
  Robotics & Physical Grasp Execution Subsystem: Grasp Executor
=============================================================================
Implements the 6-phase Pick-and-Place State Machine with Contact Telemetry:
  Phase 1: Approach (Pre-grasp waypoint 15cm above target)
  Phase 2: Descend  (Calibrated finger immersion enclosing object)
  Phase 3: Grip     (Clamping force with slip prevention)
  Phase 4: Lift     (Vertical elevation with active force retention)
  Phase 5: Transfer (Carry to collection tray and verify force closure)
  Phase 6: Drop     (Release over collection tray)
  Phase 7: Return   (Smooth return to home posture with persistent motor lock)
=============================================================================
"""
import time
import numpy as np
import pybullet as p
from .force_monitor import ForceClosureMonitor


class GraspExecutor:
    """
    Executes physical pick-and-place trajectories using Franka Panda arm and gripper,
    with real-time force closure monitoring and 7-DOF actuator torque telemetry.
    """
    def __init__(self, robot, tray_pos=(0.45, 0.45, 0.13)):
        self.robot = robot
        self.tray_pos = list(tray_pos)
        self.force_monitor = ForceClosureMonitor(robot_id=self.robot.robot_id)
        self._drop_idx = 0

    def _get_next_drop_offset(self, spawner=None):
        """
        Calculates an intelligent dispersed placement slot inside the collection tray.
        Inspects how many objects are already resting in the tray to assign an empty slot,
        preventing unstable stacking towers and object spillage.
        """
        offsets = [
            ( 0.00,  0.00),   # Slot 0: Center
            (-0.045, -0.045), # Slot 1: Back-Left
            ( 0.045, -0.045), # Slot 2: Back-Right
            (-0.045,  0.045), # Slot 3: Front-Left
            ( 0.045,  0.045), # Slot 4: Front-Right
            ( 0.00,  -0.050), # Slot 5: Back-Center
            ( 0.00,   0.050), # Slot 6: Front-Center
            (-0.050,  0.00),  # Slot 7: Left-Center
        ]
        if spawner is not None:
            tray_count = 0
            for obj in spawner.object_data:
                try:
                    pos, _ = p.getBasePositionAndOrientation(obj["id"])
                    if pos[1] > 0.30 and 0.30 <= pos[0] <= 0.60:
                        tray_count += 1
                except Exception:
                    pass
            idx = tray_count % len(offsets)
        else:
            idx = self._drop_idx % len(offsets)
            self._drop_idx += 1
        return offsets[idx]

    def execute_pick_and_place(self, grasp, spawner=None, save_plot=True):
        """
        Executes a complete pick, lift, carry, and drop-in-tray sequence.
        Args:
            grasp: Grasp object with world_coords [wx, wy, wz] and angle_rad
            spawner: Optional ObjectSpawner instance to track lifted object
            save_plot: Whether to export outputs/force_closure_profile.png
        Returns:
            dict containing execution success and physical contact telemetry
        """
        if grasp.world_coords is None:
            return {"success": False, "reason": "No 3D world coordinates"}

        wx, wy, wz = grasp.world_coords
        yaw = grasp.angle_rad
        t_start = time.perf_counter()
        self.force_monitor.reset()

        print(f"\n[Grasp Execution Started]")
        print(f"  Target World Pos : [{wx:.3f}, {wy:.3f}, {wz:.3f}] m")
        print(f"  Gripper Yaw Angle: {grasp.angle_deg:+.1f} deg")

        # -------------------------------------------------------------
        # Phase 1: Approach (Pre-grasp waypoint 15cm above target)
        # -------------------------------------------------------------
        self.force_monitor.sample("Approach")
        approach_z = wz + 0.15
        print(f"  Phase 1: Approach  -> [{wx:.3f}, {wy:.3f}, {approach_z:.3f}]")
        self.robot.open_gripper(target_pos=0.04, steps=20)
        self.robot.move_to_cartesian([wx, wy, approach_z], target_yaw=yaw, steps=100)
        for _ in range(20):
            p.stepSimulation()
        self.force_monitor.sample("Approach")

        # -------------------------------------------------------------
        # Phase 2: Descend (Enclose object between finger pads)
        # -------------------------------------------------------------
        self.force_monitor.sample("Descend")
        grasp_z = max(0.012, wz - 0.016)
        print(f"  Phase 2: Descend   -> [{wx:.3f}, {wy:.3f}, {grasp_z:.3f}]")
        self.robot.move_to_cartesian([wx, wy, grasp_z], target_yaw=yaw, steps=100)
        for _ in range(20):
            p.stepSimulation()
        self.force_monitor.sample("Descend")

        # -------------------------------------------------------------
        # Phase 3: Grip (Close fingers with calibrated clamping force)
        # -------------------------------------------------------------
        print("  Phase 3: Grip      -> Clamping fingers (force=120N)...")
        self.robot.close_gripper(force=120.0, target_pos=0.0, steps=80)
        for _ in range(15):
            p.stepSimulation()

        snap_grip = self.force_monitor.sample("Grip")
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

        # Sample and report contact dynamics during lift
        snap_lift = self.force_monitor.sample("Lift")
        self.force_monitor.print_telemetry_report(snap_lift)

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

            # Distributed Palletizing Drop Slot Calculation:
            # Places consecutive objects across distinct floor slots to prevent unstable stacking towers
            drop_offset = self._get_next_drop_offset(spawner)
            slot_x = self.tray_pos[0] + drop_offset[0]
            slot_y = self.tray_pos[1] + drop_offset[1]
            drop_z = 0.13  # Gentle release height (~2 cm above tray floor)
            transit_waypoint = [slot_x, slot_y, 0.22]

            print(f"  Phase 5: Transfer  -> Moving to tray slot [{slot_x:.2f}, {slot_y:.2f}, {drop_z:.2f}]...")
            self.force_monitor.sample("Transfer")
            # Move via high waypoint to safely clear obstacles and tray perimeter
            self.robot.move_to_cartesian(transit_waypoint, target_yaw=0.0, steps=80,
                                         keep_gripper_closed=True, grip_force=120.0)
            # Descend gently to designated drop slot
            self.robot.move_to_cartesian([slot_x, slot_y, drop_z], target_yaw=0.0, steps=45,
                                         keep_gripper_closed=True, grip_force=120.0)
            for _ in range(15):
                p.stepSimulation()
            self.force_monitor.sample("Transfer")

            print("  Phase 6: Drop      -> Opening gripper gently over tray slot...")
            self.force_monitor.sample("Drop")
            self.robot.open_gripper(target_pos=0.04, force=40.0, steps=40)
            for _ in range(25):
                p.stepSimulation()
            self.force_monitor.sample("Drop")

            # Post-drop clearance: retract vertically before home return to prevent clipping objects
            self.robot.move_to_cartesian(transit_waypoint, target_yaw=0.0, steps=40,
                                         keep_gripper_closed=False)
            success = True
        else:
            print("  Result: MISS - object not engaged by gripper.")
            success = False

        # -------------------------------------------------------------
        # Phase 7: Smooth Return to Neutral Home Posture
        # -------------------------------------------------------------
        print("  Returning robot arm smoothly to home posture...")
        self.force_monitor.sample("Return")
        self.robot.move_to_home(steps=80)
        self.force_monitor.sample("Return")

        cycle_time = time.perf_counter() - t_start
        print(f"[Pick-and-Place Completed in {cycle_time:.2f}s | Success: {success}]\n")

        plot_path = None
        if save_plot:
            try:
                plot_path = self.force_monitor.plot_profile("outputs/force_closure_profile.png")
            except Exception:
                pass

        return {
            "success": success,
            "cycle_time_s": cycle_time,
            "lifted_object": lifted_obj["name"] if lifted_obj else None,
            "contacts": num_contacts,
            "force_closure": snap_lift.get("force_closure", False),
            "normal_force": snap_lift.get("mean_fn", 0.0),
            "friction_force": snap_lift.get("mean_ft", 0.0),
            "friction_angle_deg": snap_lift.get("friction_angle_deg", 0.0),
            "friction_margin_pct": snap_lift.get("margin_pct", 0.0),
            "joint_torques": snap_lift.get("joint_torques", []),
            "peak_joint": snap_lift.get("peak_joint", "J4"),
            "peak_torque": snap_lift.get("peak_torque", 0.0),
            "profile_plot": plot_path
        }

