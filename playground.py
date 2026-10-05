#!/usr/bin/env python3
"""
=================================================================
  INTERACTIVE SIMULATION PLAYGROUND v3
=================================================================
Features:
  - 5 clearly visible, named objects from PyBullet dataset
  - Select an object → show all grasp points on it
  - Select a grasp point → execute grasp with live force readout
  - Clean minimal 3D text (no clutter)
  - Live OpenCV perception dashboard

Usage:
    python playground.py
    python playground.py --num-objects 5
    python playground.py --no-dashboard
=================================================================
"""
import argparse
import pybullet as p
import time
import numpy as np
import cv2

from simulation.environment import GraspingEnvironment
from perception.detector import GraspDetector
from perception.visualizer import Visualizer


class InteractivePlayground:
    """Interactive playground with object selection and force-aware grasping."""

    def __init__(self, num_objects=5, show_dashboard=True, checkpoint=None):
        self.show_dashboard = show_dashboard
        self.num_objects = min(num_objects, 5)

        print("=" * 65)
        print("  PICKBOT INTERACTIVE PLAYGROUND v3")
        print("  Vision-Guided Grasping with Franka Emika Panda")
        print("=" * 65)
        print()

        # 1. Initialize Environment
        print("[1/3] Initializing PyBullet simulation...")
        self.env = GraspingEnvironment(gui=True)

        # 2. Initialize Perception
        ckpt = checkpoint or "checkpoints/ggcnn_weights_cornell/ggcnn_epoch_23_cornell_statedict.pt"
        print(f"[2/3] Loading GG-CNN model: {ckpt}")
        self.detector = GraspDetector(checkpoint_path=ckpt)
        self.visualizer = Visualizer()

        # 3. Spawn objects
        print(f"[3/3] Spawning {self.num_objects} objects...")
        self.env.reset(num_objects=self.num_objects)
        for _ in range(60):
            self.env.step()

        # State
        self.last_grasps = []          # All detected grasps
        self.filtered_grasps = []      # Grasps filtered for selected object
        self.last_telemetry = {}
        self.last_heatmaps = {}
        self.grasp_markers = []        # 3D grasp visualization IDs
        self.selection_markers = []    # Object selection highlight IDs
        self.force_text_id = None      # Force readout text ID
        self.status_text_id = None     # Status text ID
        self.detection_count = 0
        self.grasp_attempts = 0
        self.successful_grasps = 0

        # Setup GUI
        self._setup_gui()
        self._update_status("Ready — select an object and click SHOW GRASP POINTS")
        self._print_help()

    # ------------------------------------------------------------------
    # GUI Setup
    # ------------------------------------------------------------------
    def _setup_gui(self):
        """Create clearly organized sliders and buttons."""
        # Robot arm control
        self.slider_x = p.addUserDebugParameter("Robot X", 0.20, 0.80, 0.50)
        self.slider_y = p.addUserDebugParameter("Robot Y", -0.40, 0.40, 0.00)
        self.slider_z = p.addUserDebugParameter("Robot Z", 0.00, 0.60, 0.35)
        self.slider_yaw = p.addUserDebugParameter("Gripper Yaw (deg)", -180, 180, 0)

        # Scene controls (integer-valued sliders)
        self.slider_num_obj = p.addUserDebugParameter("Object Count (1-5)", 1, 5, self.num_objects)

        # Object & Grasp selection (integer-valued sliders)
        self.slider_sel_obj = p.addUserDebugParameter(
            "Select Object (0=All)", 0, 5, 0
        )
        self.slider_sel_grasp = p.addUserDebugParameter(
            "Grasp Point (1-5)", 1, 5, 1
        )

        # Action buttons
        self.btn_detect = p.addUserDebugParameter(">> SHOW GRASP POINTS <<", 1, 0, 0)
        self.btn_execute = p.addUserDebugParameter(">> EXECUTE GRASP <<", 1, 0, 0)
        self.btn_respawn = p.addUserDebugParameter(">> RESPAWN OBJECTS <<", 1, 0, 0)
        self.btn_reset = p.addUserDebugParameter(">> RESET ROBOT <<", 1, 0, 0)

        # Track button states
        self.prev_detect = p.readUserDebugParameter(self.btn_detect)
        self.prev_execute = p.readUserDebugParameter(self.btn_execute)
        self.prev_respawn = p.readUserDebugParameter(self.btn_respawn)
        self.prev_reset = p.readUserDebugParameter(self.btn_reset)

    # ------------------------------------------------------------------
    # Minimal 3D Status Text
    # ------------------------------------------------------------------
    def _update_status(self, text, color=None):
        """Show a single clean status line above the workspace."""
        if color is None:
            color = [0.0, 0.85, 1.0]
        pos = [0.50, -0.28, 0.08]
        if self.status_text_id is not None:
            self.status_text_id = p.addUserDebugText(
                text, pos,
                textColorRGB=color, textSize=1.1,
                replaceItemUniqueId=self.status_text_id
            )
        else:
            self.status_text_id = p.addUserDebugText(
                text, pos,
                textColorRGB=color, textSize=1.1
            )

    def _print_help(self):
        print()
        print("=" * 65)
        print("  CONTROLS:")
        print("=" * 65)
        print("  Sliders:")
        print("    Robot X/Y/Z/Yaw : Move the robot arm")
        print("    Object Count    : How many objects to spawn (1-5)")
        print("    Select Object   : 0=All, 1-5=Specific object")
        print("    Grasp Point     : Which grasp to use (1-5)")
        print()
        print("  Buttons:")
        print("    SHOW GRASP POINTS : Detect and show grasp candidates")
        print("    EXECUTE GRASP     : Pick object at selected grasp point")
        print("    RESPAWN OBJECTS   : Clear and respawn new objects")
        print("    RESET ROBOT       : Return arm to home position")
        print()
        print("  The robot shows GRIP FORCE during grasping.")
        print("  Press Ctrl+C to exit.")
        print("=" * 65)
        print()

    # ------------------------------------------------------------------
    # Visual Markers
    # ------------------------------------------------------------------
    def _clear_grasp_markers(self):
        for mid in self.grasp_markers:
            try:
                p.removeUserDebugItem(mid)
            except Exception:
                pass
        self.grasp_markers.clear()

    def _clear_selection_markers(self):
        for mid in self.selection_markers:
            try:
                p.removeUserDebugItem(mid)
            except Exception:
                pass
        self.selection_markers.clear()

    def _clear_force_text(self):
        if self.force_text_id is not None:
            try:
                p.removeUserDebugItem(self.force_text_id)
            except Exception:
                pass
            self.force_text_id = None

    def _highlight_object(self, obj_idx):
        """Draw a selection ring around the chosen object."""
        self._clear_selection_markers()
        if obj_idx < 0 or obj_idx >= len(self.env.spawner.object_data):
            return

        data = self.env.spawner.object_data[obj_idx]
        pos = data["settled_pos"]
        radius = 0.06
        segments = 20
        z = 0.004

        for i in range(segments):
            a1 = 2 * np.pi * i / segments
            a2 = 2 * np.pi * (i + 1) / segments
            p1 = [pos[0] + radius * np.cos(a1), pos[1] + radius * np.sin(a1), z]
            p2 = [pos[0] + radius * np.cos(a2), pos[1] + radius * np.sin(a2), z]
            mid = p.addUserDebugLine(p1, p2, lineColorRGB=[0, 1, 0], lineWidth=3.0, lifeTime=0)
            self.selection_markers.append(mid)

        # Add selection arrow text
        arrow_pos = [pos[0], pos[1], pos[2] + 0.12]
        mid = p.addUserDebugText(
            f">> SELECTED: {data['name']} <<",
            arrow_pos,
            textColorRGB=[0.0, 1.0, 0.3],
            textSize=1.2
        )
        self.selection_markers.append(mid)

    def _draw_grasp_marker(self, grasp, rank, is_selected=False):
        """Draw a numbered grasp marker in the 3D scene."""
        if grasp.world_coords is None:
            return

        wx, wy, wz = grasp.world_coords

        if is_selected:
            line_color = [0.0, 1.0, 0.2]
            text_color = [0.0, 1.0, 0.3]
            line_w = 5.0
        else:
            line_color = [1.0, 0.75, 0.0]
            text_color = [1.0, 0.85, 0.2]
            line_w = 2.5

        # Vertical approach line
        m1 = p.addUserDebugLine(
            [wx, wy, wz], [wx, wy, wz + 0.12],
            lineColorRGB=line_color, lineWidth=line_w, lifeTime=0
        )
        self.grasp_markers.append(m1)

        # Gripper jaw direction bar
        angle = grasp.angle_rad
        dx = np.cos(angle) * 0.03
        dy = np.sin(angle) * 0.03
        m2 = p.addUserDebugLine(
            [wx - dx, wy - dy, wz], [wx + dx, wy + dy, wz],
            lineColorRGB=[0.0, 0.9, 1.0], lineWidth=line_w, lifeTime=0
        )
        self.grasp_markers.append(m2)

        # Crosshair
        cr = 0.015
        m3 = p.addUserDebugLine(
            [wx - cr, wy, wz], [wx + cr, wy, wz],
            lineColorRGB=line_color, lineWidth=2.0, lifeTime=0
        )
        m4 = p.addUserDebugLine(
            [wx, wy - cr, wz], [wx, wy + cr, wz],
            lineColorRGB=line_color, lineWidth=2.0, lifeTime=0
        )
        self.grasp_markers.extend([m3, m4])

        # Numbered label
        label = f"#{rank + 1}  Q={grasp.quality:.2f}"
        if is_selected:
            label = f">> #{rank + 1}  Q={grasp.quality:.2f} <<"
        m5 = p.addUserDebugText(
            label,
            [wx, wy, wz + 0.13],
            textColorRGB=text_color,
            textSize=1.1
        )
        self.grasp_markers.append(m5)

    # ------------------------------------------------------------------
    # Force Measurement
    # ------------------------------------------------------------------
    def _measure_grip_force(self):
        """Measure total normal contact force on gripper fingers."""
        total_force = 0.0
        for g_idx in self.env.robot.gripper_joints:
            contacts = p.getContactPoints(bodyA=self.env.robot.robot_id, linkIndexA=g_idx)
            for contact in contacts:
                total_force += abs(contact[9])  # normalForce
        return total_force

    def _show_force(self, wx, wy, wz, force, label="GRIP"):
        """Display force readout as 3D text near the gripper."""
        text = f"{label} FORCE: {force:.1f} N"
        pos = [wx, wy, wz + 0.22]
        color = [1.0, 0.3, 0.1] if force > 5.0 else [1.0, 0.85, 0.0]

        if self.force_text_id is not None:
            self.force_text_id = p.addUserDebugText(
                text, pos,
                textColorRGB=color, textSize=1.4,
                replaceItemUniqueId=self.force_text_id
            )
        else:
            self.force_text_id = p.addUserDebugText(
                text, pos,
                textColorRGB=color, textSize=1.4
            )

    # ------------------------------------------------------------------
    # Detection & Grasp Execution
    # ------------------------------------------------------------------
    def _find_nearest_object_z(self, wx, wy):
        """Find the actual Z height of the nearest spawned object to (wx, wy)."""
        best_z = 0.02  # fallback: just above table
        best_dist = float('inf')
        for data in self.env.spawner.object_data:
            pos = data.get("settled_pos", None)
            if pos is None:
                continue
            dist = np.sqrt((wx - pos[0])**2 + (wy - pos[1])**2)
            if dist < best_dist:
                best_dist = dist
                # Use actual object Z (from simulation), not depth-derived
                try:
                    actual_pos, _ = p.getBasePositionAndOrientation(data["id"])
                    best_z = actual_pos[2]
                except Exception:
                    best_z = float(pos[2])
        return best_z

    def run_detection(self):
        """Detect grasps and filter by selected object."""
        self.detection_count += 1
        self._clear_grasp_markers()
        self._clear_force_text()

        sel_obj = int(round(p.readUserDebugParameter(self.slider_sel_obj)))
        sel_obj = max(0, min(sel_obj, len(self.env.spawner.object_data)))

        # Highlight selected object
        if sel_obj > 0:
            self._highlight_object(sel_obj - 1)
            obj_name = self.env.spawner.object_data[sel_obj - 1]["name"]
            self._update_status(f"Detecting grasps on {obj_name}...", [0.0, 0.85, 1.0])
            print(f"\n{'─' * 60}")
            print(f"  Detection #{self.detection_count} │ Target: {obj_name}")
            print(f"{'─' * 60}")
        else:
            self._clear_selection_markers()
            self._update_status("Detecting grasps on ALL objects...", [0.0, 0.85, 1.0])
            print(f"\n{'─' * 60}")
            print(f"  Detection #{self.detection_count} │ Target: ALL objects")
            print(f"{'─' * 60}")

        # Run CNN inference
        rgb, depth, seg = self.env.get_observation()
        grasps, telemetry, heatmaps = self.detector.detect(
            rgb, depth, camera=self.env.camera, top_k=10
        )

        self.last_grasps = grasps
        self.last_telemetry = telemetry
        self.last_heatmaps = heatmaps

        # Correct grasp Z coordinates using actual object positions
        for g in grasps:
            if g.world_coords is not None:
                actual_z = self._find_nearest_object_z(g.world_coords[0], g.world_coords[1])
                g.world_coords = np.array([g.world_coords[0], g.world_coords[1], actual_z])

        # Filter by selected object
        if sel_obj > 0 and sel_obj <= len(self.env.spawner.object_data):
            obj_pos = self.env.spawner.object_data[sel_obj - 1]["settled_pos"]
            filtered = []
            for g in grasps:
                if g.world_coords is not None:
                    dist = np.linalg.norm(g.world_coords[:2] - obj_pos[:2])
                    if dist < 0.08:
                        filtered.append(g)
            self.filtered_grasps = filtered[:5]
        else:
            self.filtered_grasps = grasps[:5]

        # Print results
        print(f"  Latency    : {telemetry['inference_time_ms']:.2f} ms")
        print(f"  Device     : {telemetry['device'].upper()}")
        print(f"  Detected   : {len(grasps)} total grasps")
        print(f"  Filtered   : {len(self.filtered_grasps)} grasp points shown")
        print()

        if self.filtered_grasps:
            print(f"  {'#':>3}  {'Quality':>8}  {'Angle':>10}  {'Position (X, Y, Z)'}")
            print(f"  {'─'*3}  {'─'*8}  {'─'*10}  {'─'*28}")
            for i, g in enumerate(self.filtered_grasps):
                if g.world_coords is not None:
                    wx, wy, wz = g.world_coords
                    print(f"  {i+1:>3}  {g.quality:>8.3f}  {g.angle_deg:>+9.1f}°  [{wx:.3f}, {wy:.3f}, {wz:.3f}]")
            print()

        # Draw grasp markers in 3D
        sel_grasp = int(round(p.readUserDebugParameter(self.slider_sel_grasp))) - 1
        for rank, g in enumerate(self.filtered_grasps):
            self._draw_grasp_marker(g, rank, is_selected=(rank == sel_grasp))

        # Update status
        n = len(self.filtered_grasps)
        if n > 0:
            self._update_status(
                f"{n} grasp points found — select one and click EXECUTE GRASP",
                [0.0, 1.0, 0.3]
            )
        else:
            self._update_status("No valid grasps found. Try respawning.", [1.0, 0.4, 0.1])

        # Update OpenCV dashboard
        if self.show_dashboard:
            dashboard = self.visualizer.create_dashboard(
                rgb, depth, heatmaps, self.filtered_grasps, telemetry
            )
            cv2.imshow("Grasp Detection Dashboard", cv2.cvtColor(dashboard, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

    def execute_grasp(self):
        """Execute grasp at the selected grasp point with live force display."""
        if not self.filtered_grasps:
            print("\n  ⚠  No grasp points! Click SHOW GRASP POINTS first.")
            self._update_status("No grasp points! Click SHOW GRASP POINTS first.", [1.0, 0.4, 0.1])
            return

        sel_idx = int(round(p.readUserDebugParameter(self.slider_sel_grasp))) - 1
        sel_idx = max(0, min(sel_idx, len(self.filtered_grasps) - 1))

        grasp = self.filtered_grasps[sel_idx]
        if grasp.world_coords is None:
            print("\n  ⚠  Selected grasp has no 3D position!")
            return

        self.grasp_attempts += 1
        wx, wy, wz = grasp.world_coords
        yaw = grasp.angle_rad

        # Use actual object Z for reliable descent height
        actual_z = self._find_nearest_object_z(wx, wy)

        print(f"\n{'═' * 60}")
        print(f"  Grasp #{self.grasp_attempts} │ Point #{sel_idx + 1}")
        print(f"{'─' * 60}")
        print(f"  Target     : [{wx:.3f}, {wy:.3f}, {actual_z:.3f}] m")
        print(f"  Quality    : {grasp.quality:.3f}")
        print(f"  Angle      : {grasp.angle_deg:+.1f}°")
        print()

        # Phase 1: Approach — hover above the object
        approach_z = actual_z + 0.14
        self._update_status("Phase 1/4: Approaching target...", [1.0, 0.85, 0.1])
        print(f"  Phase 1 │ Approach → Z = {approach_z:.3f} m")
        self.env.robot.move_to_cartesian([wx, wy, approach_z], target_yaw=yaw, steps=100)
        for _ in range(25):
            self.env.step()

        # Phase 2: Descend — lower to just above the object surface
        grasp_z = actual_z + 0.015
        self._update_status("Phase 2/4: Descending to object...", [1.0, 0.70, 0.1])
        print(f"  Phase 2 │ Descend → Z = {grasp_z:.3f} m")
        self.env.robot.move_to_cartesian([wx, wy, grasp_z], target_yaw=yaw, steps=80)
        for _ in range(40):
            self.env.step()

        # Phase 3: Close gripper with force monitoring
        self._update_status("Phase 3/4: Closing gripper...", [0.0, 0.85, 1.0])
        print(f"  Phase 3 │ Closing gripper (monitoring force)...")
        for g_idx in self.env.robot.gripper_joints:
            p.setJointMotorControl2(
                self.env.robot.robot_id, g_idx,
                p.POSITION_CONTROL, targetPosition=0.0, force=45.0
            )

        max_grip_force = 0.0
        for step in range(100):
            self.env.step()
            if step % 5 == 0:
                force = self._measure_grip_force()
                max_grip_force = max(max_grip_force, force)
                self._show_force(wx, wy, actual_z, force, label="GRIP")

        print(f"          │ Peak grip force : {max_grip_force:.1f} N")

        # Phase 4: Lift
        lift_z = actual_z + 0.22
        self._update_status(f"Phase 4/4: Lifting (Force: {max_grip_force:.1f}N)...", [0.7, 0.3, 1.0])
        print(f"  Phase 4 │ Lift     → Z = {lift_z:.3f} m")
        self.env.robot.move_to_cartesian([wx, wy, lift_z], target_yaw=yaw, steps=100)

        max_lift_force = 0.0
        for step in range(80):
            self.env.step()
            if step % 5 == 0:
                force = self._measure_grip_force()
                max_lift_force = max(max_lift_force, force)
                self._show_force(wx, wy, lift_z, force, label="LIFT")

        print(f"          │ Peak lift force : {max_lift_force:.1f} N")
        print()

        # Check success — any object raised above 0.09m
        grasped = False
        grasped_name = ""
        for data in self.env.spawner.object_data:
            try:
                pos, _ = p.getBasePositionAndOrientation(data["id"])
                if pos[2] > 0.09:
                    grasped = True
                    grasped_name = data["name"]
                    break
            except Exception:
                pass

        if grasped:
            self.successful_grasps += 1
            print(f"  Result  │ ✓ SUCCESS — {grasped_name} lifted!")
            print(f"          │ Grip: {max_grip_force:.1f} N  │  Lift: {max_lift_force:.1f} N")
            self._update_status(
                f"SUCCESS! Grip: {max_grip_force:.1f}N, Lift: {max_lift_force:.1f}N",
                [0.0, 1.0, 0.3]
            )

            # Transfer to tray
            print(f"          │ Transferring to collection tray...")
            self.env.robot.move_to_cartesian([0.50, 0.40, 0.22], target_yaw=0.0, steps=100)
            for _ in range(30):
                self.env.step()

            # Release
            for g_idx in self.env.robot.gripper_joints:
                p.setJointMotorControl2(
                    self.env.robot.robot_id, g_idx,
                    p.POSITION_CONTROL, targetPosition=0.04, force=20.0
                )
            for _ in range(40):
                self.env.step()
        else:
            print(f"  Result  │ ✗ MISS — object not grasped")
            print(f"          │ Grip: {max_grip_force:.1f} N (insufficient contact)")
            self._update_status(
                f"MISS — Grip: {max_grip_force:.1f}N (insufficient contact)",
                [1.0, 0.3, 0.1]
            )

        rate = 100 * self.successful_grasps / max(1, self.grasp_attempts)
        print(f"{'─' * 60}")
        print(f"  Stats   │ {self.successful_grasps}/{self.grasp_attempts} successful ({rate:.0f}%)")
        print(f"{'═' * 60}")

        # Return home
        print("\n  Returning robot to home position...")
        self.env.robot.reset()
        for _ in range(30):
            self.env.step()
        self._clear_force_text()

    def respawn_objects(self):
        """Clear and respawn with current slider count."""
        num = int(round(p.readUserDebugParameter(self.slider_num_obj)))
        num = max(1, min(num, 5))
        print(f"\n{'─' * 60}")
        print(f"  Respawn │ Spawning {num} object{'s' if num > 1 else ''}...")
        print(f"{'─' * 60}")
        self._clear_grasp_markers()
        self._clear_selection_markers()
        self._clear_force_text()
        self.filtered_grasps = []
        self.last_grasps = []
        self.env.reset(num_objects=num)
        for _ in range(60):
            self.env.step()
        obj_info = self.env.spawner.get_object_info()
        for i, o in enumerate(obj_info, 1):
            print(f"    {i}. {o['name']}")
        print()
        self._update_status("Objects respawned — select one and click SHOW GRASP POINTS")

    # ------------------------------------------------------------------
    # Main Loop
    # ------------------------------------------------------------------
    def run(self):
        """Main interactive loop."""
        print(f"\n{'═' * 60}")
        print(f"  Playground active │ Press Ctrl+C to exit")
        print(f"{'═' * 60}\n")

        if self.show_dashboard:
            cv2.namedWindow("Grasp Detection Dashboard", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Grasp Detection Dashboard", 800, 520)

        try:
            while True:
                # Check if PyBullet is still connected
                if not p.isConnected():
                    print("\n  PyBullet window closed.")
                    break

                # Read robot sliders
                tx = p.readUserDebugParameter(self.slider_x)
                ty = p.readUserDebugParameter(self.slider_y)
                tz = p.readUserDebugParameter(self.slider_z)
                tyaw = np.radians(p.readUserDebugParameter(self.slider_yaw))
                self.env.robot.move_to_cartesian([tx, ty, tz], target_yaw=tyaw, steps=1)

                # Update grasp selection highlighting
                if self.filtered_grasps:
                    sel = int(round(p.readUserDebugParameter(self.slider_sel_grasp))) - 1
                    # Redraw markers periodically is expensive, skip for now
                    # (markers are drawn on detection)

                # Check buttons
                curr = p.readUserDebugParameter(self.btn_detect)
                if curr != self.prev_detect:
                    self.prev_detect = curr
                    self.run_detection()

                curr = p.readUserDebugParameter(self.btn_execute)
                if curr != self.prev_execute:
                    self.prev_execute = curr
                    self.execute_grasp()

                curr = p.readUserDebugParameter(self.btn_respawn)
                if curr != self.prev_respawn:
                    self.prev_respawn = curr
                    self.respawn_objects()

                curr = p.readUserDebugParameter(self.btn_reset)
                if curr != self.prev_reset:
                    self.prev_reset = curr
                    print(f"\n  Reset │ Returning robot to home position.")
                    self.env.robot.reset()
                    self._update_status("Robot reset to home position.")

                self.env.step()

                if self.show_dashboard:
                    cv2.waitKey(1)

                time.sleep(0.005)

        except KeyboardInterrupt:
            print("\n\n  Shutting down...")
        except Exception:
            pass
        finally:
            if self.show_dashboard:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass
            try:
                self.env.close()
            except Exception:
                pass
            print("\n  Goodbye!\n")


def main():
    parser = argparse.ArgumentParser(description="PickBot Interactive Playground")
    parser.add_argument("--num-objects", type=int, default=5, help="Number of objects (1-5)")
    parser.add_argument("--no-dashboard", action="store_true", help="Disable OpenCV dashboard")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/ggcnn_weights_cornell/ggcnn_epoch_23_cornell_statedict.pt",
                        help="Path to model weights")
    args = parser.parse_args()

    playground = InteractivePlayground(
        num_objects=args.num_objects,
        show_dashboard=not args.no_dashboard,
        checkpoint=args.checkpoint,
    )
    playground.run()


if __name__ == "__main__":
    main()
