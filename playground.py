#!/usr/bin/env python3
"""
=============================================================================
  CYBER PHYSICAL SYSTEMS: INTERACTIVE SIMULATION PLAYGROUND (EVAL 2)
  Autonomous Vision-Guided Robotic Grasping & Pick-and-Place System
=============================================================================
Features:
  - PyBullet 3D GUI with studio lighting, shadow mapping, and 3D HUD banner.
  - GG-CNN real-time grasp perception running on CUDA GPU.
  - 4-Panel OpenCV Live Perception Dashboard.
  - Single Pick execution with automatic re-targeting (no repeat picking).
  - Autonomous "Pick All (Clear Scene)" & "Batch Pick (Count)" without stopping.
  - Integer-clamped object counts with dynamic 3D configuration display.
  - Calibrated collection tray transfer with zero arm drift on return home.
=============================================================================
"""
import argparse
import time
import sys
import os
import numpy as np
import cv2
import pybullet as p

from simulation.environment import GraspingEnvironment
from simulation.grasp_executor import GraspExecutor
from perception.detector import GraspDetector
from perception.visualizer import Visualizer


class InteractivePlayground:
    """Full interactive simulation playground with PyBullet GUI + OpenCV HUD."""

    def __init__(self, num_objects=3, show_dashboard=True, checkpoint=None):
        self.show_dashboard = show_dashboard
        self.num_objects = max(1, min(8, int(round(num_objects))))

        print("=" * 75)
        print("  CYBER PHYSICAL SYSTEMS: INTERACTIVE PICK-AND-PLACE PLAYGROUND")
        print("  Autonomous AI-Based Robotic Manipulation System")
        print("=" * 75)
        print()

        # 1. Initialize PyBullet Simulation Environment
        print("[1/3] Initializing PyBullet Simulation with GUI...")
        self.env = GraspingEnvironment(gui=True)

        # 2. Initialize Perception Model
        ckpt = checkpoint or "checkpoints/ggcnn_weights_cornell/ggcnn_epoch_23_cornell_statedict.pt"
        if not os.path.exists(ckpt):
            ckpt = "checkpoints/ggcnn_finetuned.pt"
        print(f"[2/3] Loading GG-CNN Perception Model: {ckpt}")
        self.detector = GraspDetector(checkpoint_path=ckpt)
        self.visualizer = Visualizer()
        self.executor = GraspExecutor(self.env.robot, tray_pos=(0.45, 0.45, 0.22))

        # 3. Spawn initial calibrated scene
        print(f"[3/3] Spawning {self.num_objects} objects in workspace...")
        self.env.reset(num_objects=self.num_objects)
        for _ in range(60):
            self.env.step()

        # State tracking
        self.last_grasps = []
        self.last_telemetry = {}
        self.last_heatmaps = {}
        self.last_force_snapshot = {}
        self.grasp_markers = []
        self.hud_text_ids = []
        self.detection_count = 0
        self.grasp_attempts = 0
        self.successful_grasps = 0
        self.status_message = "READY"

        # Multi-View Dynamic 3D Camera Presets
        self.camera_presets = [
            ("Isometric Studio (3D)", 1.05, 48.0, -28.0, [0.48, 0.08, 0.12]),
            ("Over-The-Shoulder (OTS)", 0.88, 135.0, -24.0, [0.48, 0.05, 0.15]),
            ("Top-Down Inspection (90°)", 0.88, 0.0, -89.9, [0.50, 0.00, 0.02]),
            ("Side Clearance Profile", 0.92, 90.0, -12.0, [0.48, 0.05, 0.08])
        ]
        self.cam_idx = 0

        # Setup GUI controls
        self._setup_gui_controls()
        self._draw_tray_marker()
        self._update_3d_hud()
        self._print_help()

    def _setup_gui_controls(self):
        """Create PyBullet GUI sliders and action buttons."""
        # Section Header Text in 3D Scene
        p.addUserDebugText(
            "✦ CPS PICKBOT STUDIO ✦",
            [0.50, 0.00, 0.58],
            textColorRGB=[0.08, 0.88, 0.92],
            textSize=1.4
        )

        # Action Buttons (Sacrificial '>>' prefix absorbs OpenGL 3 vertex 0 quad artifact)
        self.btn_detect = p.addUserDebugParameter(">> [1] DETECT GRASPS <<", 1, 0, 0)
        self.btn_execute = p.addUserDebugParameter(">> [2] EXECUTE NEXT GRASP <<", 1, 0, 0)
        self.btn_pick_all = p.addUserDebugParameter(">> [3] PICK ALL (CLEAR SCENE) <<", 1, 0, 0)
        self.btn_batch_pick = p.addUserDebugParameter(">> [4] PICK BATCH (COUNT) <<", 1, 0, 0)
        self.btn_respawn = p.addUserDebugParameter(">> [5] RESPAWN OBJECTS <<", 1, 0, 0)
        self.btn_reset_robot = p.addUserDebugParameter(">> [6] RESET ROBOT TO HOME <<", 1, 0, 0)
        self.btn_cam = p.addUserDebugParameter(">> [7] SWITCH CAMERA VIEW <<", 1, 0, 0)

        # Configuration Sliders (Prefix prevents first letter of words from ever being clipped)
        self.slider_num_obj = p.addUserDebugParameter(">> Spawn Count [1 to 8]", 1, 8, self.num_objects)
        self.slider_batch_count = p.addUserDebugParameter(">> Batch Pick Count [1 to 8]", 1, 8, 2)

        # Manual Arm Controls (X, Y, Z, Yaw)
        self.slider_x = p.addUserDebugParameter(">> Manual Arm X", 0.20, 0.80, 0.50)
        self.slider_y = p.addUserDebugParameter(">> Manual Arm Y", -0.40, 0.40, 0.00)
        self.slider_z = p.addUserDebugParameter(">> Manual Arm Z", 0.05, 0.60, 0.35)
        self.slider_yaw = p.addUserDebugParameter(">> Manual Gripper Yaw", -180, 180, 0)

        # Store previous button values to detect user clicks
        self.prev_detect = p.readUserDebugParameter(self.btn_detect)
        self.prev_execute = p.readUserDebugParameter(self.btn_execute)
        self.prev_pick_all = p.readUserDebugParameter(self.btn_pick_all)
        self.prev_batch_pick = p.readUserDebugParameter(self.btn_batch_pick)
        self.prev_respawn = p.readUserDebugParameter(self.btn_respawn)
        self.prev_reset_robot = p.readUserDebugParameter(self.btn_reset_robot)
        self.prev_cam = p.readUserDebugParameter(self.btn_cam)

    def set_camera_view(self, idx=None):
        """Switches PyBullet 3D viewport between calibrated camera presets."""
        if idx is None:
            self.cam_idx = (self.cam_idx + 1) % len(self.camera_presets)
        else:
            self.cam_idx = idx % len(self.camera_presets)
        name, dist, yaw, pitch, target = self.camera_presets[self.cam_idx]
        p.resetDebugVisualizerCamera(
            cameraDistance=dist,
            cameraYaw=yaw,
            cameraPitch=pitch,
            cameraTargetPosition=target
        )
        print(f"\n[Camera Controller] Active View: {name}")
        self.status_message = f"VIEW: {name.split()[0]}"
        self._update_3d_hud()

    def _draw_tray_marker(self):
        """Draws a visual boundary box around the collection drop tray."""
        tx, ty, tz = 0.45, 0.45, 0.002
        half_w = 0.11
        p.addUserDebugLine([tx - half_w, ty - half_w, tz], [tx + half_w, ty - half_w, tz], [0.95, 0.78, 0.10], 2.0)
        p.addUserDebugLine([tx + half_w, ty - half_w, tz], [tx + half_w, ty + half_w, tz], [0.95, 0.78, 0.10], 2.0)
        p.addUserDebugLine([tx + half_w, ty + half_w, tz], [tx - half_w, ty + half_w, tz], [0.95, 0.78, 0.10], 2.0)
        p.addUserDebugLine([tx - half_w, ty + half_w, tz], [tx - half_w, ty - half_w, tz], [0.95, 0.78, 0.10], 2.0)
        # Position label on open table surface in front of yellow border to prevent intersecting the tray mesh
        p.addUserDebugText("COLLECTION TRAY", [tx - 0.08, ty - 0.18, tz + 0.005], [0.95, 0.78, 0.10], 0.85)

    def _update_3d_hud(self):
        """Updates the high-visibility floating 3D status HUD in PyBullet."""
        for txt_id in self.hud_text_ids:
            try:
                p.removeUserDebugItem(txt_id)
            except Exception:
                pass
        self.hud_text_ids.clear()

        table_objs = self.get_table_objects()
        tray_objs = self.get_tray_objects()
        spawn_n = int(round(p.readUserDebugParameter(self.slider_num_obj)))
        batch_n = int(round(p.readUserDebugParameter(self.slider_batch_count)))

        rate_str = f"{100.0 * self.successful_grasps / max(1, self.grasp_attempts):.0f}%" if self.grasp_attempts > 0 else "N/A"

        line1 = f"Status: [{self.status_message}] | Table: {len(table_objs)} objs | Tray: {len(tray_objs)} objs"
        line2 = f"Success: {rate_str} ({self.successful_grasps}/{self.grasp_attempts}) | Config: Spawn [{spawn_n}] | Batch [{batch_n}]"

        t1 = p.addUserDebugText(line1, [0.50, 0.00, 0.53], textColorRGB=[1.0, 1.0, 1.0], textSize=1.0)
        t2 = p.addUserDebugText(line2, [0.50, 0.00, 0.48], textColorRGB=[0.75, 0.85, 0.95], textSize=0.9)
        self.hud_text_ids.extend([t1, t2])

        # Line 3: Live Cyber-Physical Force Closure & Actuator Torque in 3D Scene
        if self.last_force_snapshot:
            fn = self.last_force_snapshot.get("normal_force", 0.0)
            th = self.last_force_snapshot.get("friction_angle_deg", 0.0)
            margin = self.last_force_snapshot.get("friction_margin_pct", 0.0)
            pj = self.last_force_snapshot.get("peak_joint", "J2")
            pt = self.last_force_snapshot.get("peak_torque", 0.0)
            fc_str = "ACHIEVED" if self.last_force_snapshot.get("force_closure") else "PENDING"
            line3 = f"Force Closure: [{fc_str}] | F_normal: {fn:.1f}N | Angle θ: {th:.1f}° | Margin: {margin:.0f}% | Peak Torque: {pj}={pt:+.1f}Nm"
            t3 = p.addUserDebugText(line3, [0.50, 0.00, 0.43], textColorRGB=[0.0, 1.0, 0.6], textSize=0.88)
            self.hud_text_ids.append(t3)

    def _print_help(self):
        """Print interactive control commands."""
        print()
        print("=" * 75)
        print("  INTERACTIVE GUI CONTROLS & CAMERA PRESETS:")
        print("=" * 75)
        print("  [Action Buttons - Right Panel]")
        print("    >> [1] DETECT GRASPS       : Run GG-CNN inference on remaining objects")
        print("    >> [2] EXECUTE NEXT GRASP  : Pick next object into tray (auto-refreshes)")
        print("    >> [3] PICK ALL (AUTONOMOUS): Pick all objects one-by-one until scene is clear")
        print("    >> [4] PICK BATCH (COUNT)  : Pick exact count set by 'Batch Pick Count' slider")
        print("    >> [5] RESPAWN OBJECTS     : Clear scene and spawn new randomized objects")
        print("    >> [6] RESET ROBOT TO HOME : Smoothly return arm to tucked home posture")
        print("    >> [7] SWITCH CAMERA VIEW  : Cycle 3D camera (Isometric / OTS / Overhead / Side)")
        print()
        print("  [Keyboard Shortcuts]")
        print("    'c' or 'C' : Cycle 3D camera view")
        print("    '1' to '4' : Direct jump to camera presets:")
        print("                 1 = Isometric Studio | 2 = Over-The-Shoulder (OTS)")
        print("                 3 = Top-Down Overhead | 4 = Side Clearance Profile")
        print("    'd' or 'D' : Trigger grasp detection")
        print("    'e' or 'E' : Execute next pick-and-place")
        print("    'r' or 'R' : Reset robot arm to home posture")
        print("    'q'        : Quit application")
        print()
        print("  [OpenCV Dashboard Window]")
        print("    Displays live 4-panel RGB, Depth, Quality Heatmap, and Angle Map.")
        print()
        print("  Press Ctrl+C in terminal or 'q' in OpenCV window to exit.")
        print("=" * 75)
        print()

    def get_table_objects(self):
        """Returns list of spawned objects currently resting on the picking table."""
        table_objs = []
        for obj in self.env.spawner.object_data:
            try:
                pos, _ = p.getBasePositionAndOrientation(obj["id"])
                # Table workspace: X in [0.30, 0.70], Y in [-0.22, 0.22], Z < 0.10
                if 0.30 <= pos[0] <= 0.70 and -0.22 <= pos[1] <= 0.22 and pos[2] < 0.10:
                    table_objs.append((obj, pos))
            except Exception:
                pass
        return table_objs

    def get_tray_objects(self):
        """Returns list of spawned objects delivered into the collection tray."""
        tray_objs = []
        for obj in self.env.spawner.object_data:
            try:
                pos, _ = p.getBasePositionAndOrientation(obj["id"])
                # Tray location: Y > 0.30 and X in [0.30, 0.60]
                if pos[1] > 0.30 and 0.30 <= pos[0] <= 0.60:
                    tray_objs.append((obj, pos))
            except Exception:
                pass
        return tray_objs

    def _clear_grasp_markers(self):
        """Remove debug markers from the 3D scene."""
        for m in self.grasp_markers:
            try:
                p.removeUserDebugItem(m)
            except Exception:
                pass
        self.grasp_markers.clear()

    def _draw_grasp_in_sim(self, grasp, rank=0):
        """Draws clean, professional antipodal grasp markers in the 3D PyBullet scene."""
        if grasp.world_coords is None:
            return

        wx, wy, wz = grasp.world_coords
        is_top = (rank == 0)
        color = [0.0, 1.0, 0.4] if is_top else [1.0, 0.75, 0.10]
        lw = 3.5 if is_top else 2.0

        # 1. Clean vertical approach guide line
        m1 = p.addUserDebugLine([wx, wy, wz + 0.005], [wx, wy, wz + 0.10], lineColorRGB=color, lineWidth=lw, lifeTime=0)
        self.grasp_markers.append(m1)

        # 2. Parallel antipodal finger contact pads
        yaw = grasp.angle_rad
        span = 0.040 / 2.0  # 40mm jaw span
        pad_len = 0.012

        # Closing axis vector u and perpendicular pad vector v
        u_x, u_y = np.cos(yaw), np.sin(yaw)
        v_x, v_y = -np.sin(yaw), np.cos(yaw)

        # Left pad line (cyan)
        lp_x, lp_y = wx + span * u_x, wy + span * u_y
        m2 = p.addUserDebugLine(
            [lp_x - pad_len * v_x, lp_y - pad_len * v_y, wz + 0.005],
            [lp_x + pad_len * v_x, lp_y + pad_len * v_y, wz + 0.005],
            lineColorRGB=[0.08, 0.88, 0.92], lineWidth=3.2, lifeTime=0
        )
        # Right pad line (cyan)
        rp_x, rp_y = wx - span * u_x, wy - span * u_y
        m3 = p.addUserDebugLine(
            [rp_x - pad_len * v_x, rp_y - pad_len * v_y, wz + 0.005],
            [rp_x + pad_len * v_x, rp_y + pad_len * v_y, wz + 0.005],
            lineColorRGB=[0.08, 0.88, 0.92], lineWidth=3.2, lifeTime=0
        )
        # Connecting span bar
        m4 = p.addUserDebugLine([lp_x, lp_y, wz + 0.005], [rp_x, rp_y, wz + 0.005],
                                lineColorRGB=color, lineWidth=1.8, lifeTime=0)
        self.grasp_markers.extend([m2, m3, m4])

        # 3. Clean floating target badge at top of stem (only top 2 candidates)
        if rank < 2:
            badge_text = f"#{rank+1}: Q={grasp.quality:.2f}"
            m5 = p.addUserDebugText(badge_text, [wx, wy, wz + 0.11], textColorRGB=[1.0, 1.0, 1.0], textSize=0.88, lifeTime=0)
            self.grasp_markers.append(m5)

    def run_detection(self):
        """Runs GG-CNN perception on the remaining table objects."""
        self.detection_count += 1
        self._clear_grasp_markers()

        # Guarantee camera view is 100% unobstructed with zero robot arm pixels
        self.env.robot.reset()
        for _ in range(15):
            self.env.step()

        self.status_message = "PERCEPTION SCANNING..."
        self._update_3d_hud()

        print(f"\n[Detection #{self.detection_count}] Running GG-CNN inference...")

        rgb, depth, seg = self.env.get_observation()
        grasps, telemetry, heatmaps = self.detector.detect(
            rgb, depth, camera=self.env.camera, top_k=5
        )

        # Filter grasps to ensure they strictly lie on the active picking table (Z < 0.12m)
        valid_grasps = [g for g in grasps if g.world_coords is not None and g.world_coords[2] < 0.12]

        self.last_grasps = valid_grasps
        if self.last_force_snapshot:
            telemetry.update({
                "force_closure": self.last_force_snapshot.get("force_closure"),
                "normal_force": self.last_force_snapshot.get("normal_force", 18.5),
                "friction_margin_pct": self.last_force_snapshot.get("friction_margin_pct", 87.5),
                "peak_joint": self.last_force_snapshot.get("peak_joint", "J4"),
                "peak_torque": self.last_force_snapshot.get("peak_torque", 18.6)
            })
        self.last_telemetry = telemetry
        self.last_heatmaps = heatmaps

        table_objs = self.get_table_objects()
        print(f"  Latency    : {telemetry['inference_time_ms']:.2f} ms ({telemetry['fps']:.0f} FPS)")
        print(f"  Device     : {telemetry['device'].upper()}")
        print(f"  Objects    : {len(table_objs)} remaining on table")
        print(f"  Grasp Peaks: {len(valid_grasps)} candidates detected")

        if valid_grasps:
            top = valid_grasps[0]
            wx, wy, wz = top.world_coords
            print(f"  Top Grasp  : Q={top.quality:.3f} | Angle: {top.angle_deg:+.1f}deg | Pos: [{wx:.3f}, {wy:.3f}, {wz:.3f}]")
            for rank, g in enumerate(valid_grasps[:4]):
                self._draw_grasp_in_sim(g, rank=rank)
            self.status_message = f"TARGET ACQUIRED (Q={top.quality:.2f})"
        else:
            print("  No grasp candidates found on table surface.")
            self.status_message = "NO OBJECTS DETECTED"

        self._update_3d_hud()

        # Update OpenCV dashboard
        if self.show_dashboard and heatmaps:
            dashboard = self.visualizer.create_dashboard(rgb, depth, heatmaps, valid_grasps, telemetry)
            cv2.imshow("CPS Grasp Detection Dashboard", cv2.cvtColor(dashboard, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        return valid_grasps

    def execute_single_grasp(self):
        """
        Executes a pick on the top candidate, drops it in the tray,
        and automatically re-scans the table for the next object.
        """
        table_objs = self.get_table_objects()
        if not table_objs:
            print("\n[Execute] No objects remaining on table! All objects are in tray.")
            self.status_message = "ALL OBJECTS IN TRAY (CLICK RESPAWN)"
            self._update_3d_hud()
            return False

        # If no grasp stored or scene changed, run fresh detection
        if not self.last_grasps:
            self.run_detection()
            if not self.last_grasps:
                print("[Execute] No valid grasps detected on table.")
                return False

        top = self.last_grasps[0]
        self.grasp_attempts += 1
        self.status_message = f"EXECUTING PICK #{self.grasp_attempts}..."
        self._update_3d_hud()

        print(f"\n{'='*70}")
        print(f"  EXECUTE PICK-AND-PLACE (Attempt #{self.grasp_attempts})")
        print(f"{'='*70}")

        result = self.executor.execute_pick_and_place(top, spawner=self.env.spawner)

        # Merge physical force closure & joint dynamics telemetry
        if "force_closure" in result:
            self.last_force_snapshot = result
            self.last_telemetry.update({
                "force_closure": result["force_closure"],
                "normal_force": result.get("normal_force", 18.5),
                "friction_margin_pct": result.get("friction_margin_pct", 87.5),
                "peak_joint": result.get("peak_joint", "J4"),
                "peak_torque": result.get("peak_torque", 18.6)
            })

        if result["success"]:
            self.successful_grasps += 1
            lifted = result.get("lifted_object", "Object")
            fc_str = " | FC: PASS" if result.get("force_closure") else ""
            self.status_message = f"DELIVERED [{lifted}] TO TRAY{fc_str}"
            print(f"  >>> PICK-AND-PLACE SUCCESSFUL! Delivered {lifted} to tray. <<<")
        else:
            self.status_message = "GRASP MISSED"
            print(f"  >>> GRASP MISSED - Object not secured in fingers. <<<")

        self._update_3d_hud()

        # Immediately run fresh perception scan on the remaining objects
        # so next click automatically targets the next object!
        rem_objs = self.get_table_objects()
        if rem_objs:
            print(f"\n--> Auto-scanning table ({len(rem_objs)} objects remaining)...")
            self.run_detection()
        else:
            self.last_grasps.clear()
            self._clear_grasp_markers()
            self.status_message = "SCENE CLEARED! ALL IN TRAY"
            self._update_3d_hud()

        return result["success"]

    def batch_pick(self, target_count=None):
        """
        Autonomously picks objects one by one into the tray without user stopping.
        target_count: Number of objects to pick. If None, picks ALL objects.
        """
        table_objs = self.get_table_objects()
        if not table_objs:
            print("\n[Batch Pick] Table is already empty! Click RESPAWN to spawn objects.")
            self.status_message = "TABLE EMPTY (CLICK RESPAWN)"
            self._update_3d_hud()
            return

        total_to_pick = len(table_objs) if target_count is None else min(len(table_objs), int(target_count))
        mode_name = "PICK ALL (CLEAR SCENE)" if target_count is None else f"BATCH PICK ({total_to_pick} OBJECTS)"

        print(f"\n{'='*75}")
        print(f"  STARTING AUTONOMOUS {mode_name}")
        print(f"  Target: Pick {total_to_pick} objects one by one into the collection tray")
        print(f"{'='*75}")

        picked_count = 0
        for i in range(total_to_pick):
            rem = self.get_table_objects()
            if not rem:
                print("\n[Batch Pick] All table objects cleared!")
                break

            print(f"\n--> [Batch Step {i+1}/{total_to_pick}] Objects on table: {len(rem)}")
            self.status_message = f"BATCH {i+1}/{total_to_pick}: SCANNING..."
            self._update_3d_hud()

            grasps = self.run_detection()
            if not grasps:
                print("  No grasp candidate found. Ending batch.")
                break

            top = grasps[0]
            self.grasp_attempts += 1
            self.status_message = f"BATCH {i+1}/{total_to_pick}: GRASPING..."
            self._update_3d_hud()

            res = self.executor.execute_pick_and_place(top, spawner=self.env.spawner)
            if res["success"]:
                self.successful_grasps += 1
                picked_count += 1
                print(f"  [Step {i+1}/{total_to_pick}] Successfully delivered to tray.")
            else:
                print(f"  [Step {i+1}/{total_to_pick}] Missed. Retrying next candidate...")

            self._update_3d_hud()

        print(f"\n{'='*75}")
        print(f"  {mode_name} FINISHED: {picked_count}/{total_to_pick} objects delivered to tray.")
        print(f"{'='*75}\n")

        self.status_message = f"BATCH COMPLETE: {picked_count}/{total_to_pick} DELIVERED"
        self._update_3d_hud()

        # Final fresh scan
        if self.get_table_objects():
            self.run_detection()
        else:
            self.last_grasps.clear()
            self._clear_grasp_markers()

    def respawn_objects(self):
        """Clear scene and spawn randomized objects matching the integer slider count."""
        num = int(round(p.readUserDebugParameter(self.slider_num_obj)))
        num = max(1, min(8, num))
        print(f"\n[Respawn] Spawning {num} new calibrated objects...")
        self._clear_grasp_markers()
        self.last_grasps = []
        self.env.reset(num_objects=num)
        for _ in range(60):
            self.env.step()
        self.status_message = f"RESPAWNED {num} OBJECTS"
        self._update_3d_hud()
        obj_info = self.env.spawner.get_object_info()
        print(f"  Spawned: {', '.join(o['name'] for o in obj_info)}")

    def run(self):
        """Main interactive loop."""
        print("\n[Playground] Active! Use GUI controls on the right panel or press Ctrl+C to exit.\n")

        if self.show_dashboard:
            cv2.namedWindow("CPS Grasp Detection Dashboard", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("CPS Grasp Detection Dashboard", 820, 520)

        # Track previous slider values to only move robot on explicit slider adjustment
        prev_slider_vals = (
            p.readUserDebugParameter(self.slider_x),
            p.readUserDebugParameter(self.slider_y),
            p.readUserDebugParameter(self.slider_z),
            p.readUserDebugParameter(self.slider_yaw)
        )
        prev_spawn_n = int(round(p.readUserDebugParameter(self.slider_num_obj)))
        prev_batch_n = int(round(p.readUserDebugParameter(self.slider_batch_count)))

        try:
            while True:
                # 1. Check if configuration sliders changed to update 3D HUD
                curr_spawn_n = int(round(p.readUserDebugParameter(self.slider_num_obj)))
                curr_batch_n = int(round(p.readUserDebugParameter(self.slider_batch_count)))
                if curr_spawn_n != prev_spawn_n or curr_batch_n != prev_batch_n:
                    prev_spawn_n = curr_spawn_n
                    prev_batch_n = curr_batch_n
                    self._update_3d_hud()

                # 2. Check manual arm sliders
                tx = p.readUserDebugParameter(self.slider_x)
                ty = p.readUserDebugParameter(self.slider_y)
                tz = p.readUserDebugParameter(self.slider_z)
                tyaw_deg = p.readUserDebugParameter(self.slider_yaw)
                tyaw = np.radians(tyaw_deg)
                curr_sliders = (tx, ty, tz, tyaw_deg)

                if any(abs(c - p_val) > 0.005 for c, p_val in zip(curr_sliders, prev_slider_vals)):
                    prev_slider_vals = curr_sliders
                    self.env.robot.move_to_cartesian([tx, ty, tz], target_yaw=tyaw, steps=1)

                # 3. Check Action Button Clicks
                # [1] Detect Grasps
                curr_detect = p.readUserDebugParameter(self.btn_detect)
                if curr_detect != self.prev_detect:
                    self.prev_detect = curr_detect
                    self.run_detection()

                # [2] Execute Single Next Grasp
                curr_execute = p.readUserDebugParameter(self.btn_execute)
                if curr_execute != self.prev_execute:
                    self.prev_execute = curr_execute
                    self.execute_single_grasp()

                # [3] Pick All (Clear Scene)
                curr_pick_all = p.readUserDebugParameter(self.btn_pick_all)
                if curr_pick_all != self.prev_pick_all:
                    self.prev_pick_all = curr_pick_all
                    self.batch_pick(target_count=None)

                # [4] Pick Batch (Specified Count)
                curr_batch_pick = p.readUserDebugParameter(self.btn_batch_pick)
                if curr_batch_pick != self.prev_batch_pick:
                    self.prev_batch_pick = curr_batch_pick
                    batch_target = int(round(p.readUserDebugParameter(self.slider_batch_count)))
                    self.batch_pick(target_count=batch_target)

                # [5] Respawn Objects
                curr_respawn = p.readUserDebugParameter(self.btn_respawn)
                if curr_respawn != self.prev_respawn:
                    self.prev_respawn = curr_respawn
                    self.respawn_objects()

                # [6] Reset Robot
                curr_reset = p.readUserDebugParameter(self.btn_reset_robot)
                if curr_reset != self.prev_reset_robot:
                    self.prev_reset_robot = curr_reset
                    print("\n[Reset] Returning robot arm smoothly to home posture...")
                    self.env.robot.move_to_home(steps=60)
                    prev_slider_vals = (
                        p.readUserDebugParameter(self.slider_x),
                        p.readUserDebugParameter(self.slider_y),
                        p.readUserDebugParameter(self.slider_z),
                        p.readUserDebugParameter(self.slider_yaw)
                    )
                    self.status_message = "ROBOT AT HOME"
                    self._update_3d_hud()

                # [7] Switch 3D Camera View
                curr_cam = p.readUserDebugParameter(self.btn_cam)
                if curr_cam != self.prev_cam:
                    self.prev_cam = curr_cam
                    self.set_camera_view()

                # 4. Check Keyboard Shortcuts in PyBullet GUI
                keys = p.getKeyboardEvents()
                if (ord('c') in keys and (keys[ord('c')] & p.KEY_WAS_TRIGGERED)) or \
                   (ord('C') in keys and (keys[ord('C')] & p.KEY_WAS_TRIGGERED)):
                    self.set_camera_view()
                for k_i in range(min(4, len(self.camera_presets))):
                    k_code = ord(str(k_i + 1))
                    if k_code in keys and (keys[k_code] & p.KEY_WAS_TRIGGERED):
                        self.set_camera_view(k_i)
                if (ord('d') in keys and (keys[ord('d')] & p.KEY_WAS_TRIGGERED)) or \
                   (ord('D') in keys and (keys[ord('D')] & p.KEY_WAS_TRIGGERED)):
                    self.run_detection()
                if (ord('e') in keys and (keys[ord('e')] & p.KEY_WAS_TRIGGERED)) or \
                   (ord('E') in keys and (keys[ord('E')] & p.KEY_WAS_TRIGGERED)):
                    self.execute_single_grasp()
                if (ord('r') in keys and (keys[ord('r')] & p.KEY_WAS_TRIGGERED)) or \
                   (ord('R') in keys and (keys[ord('R')] & p.KEY_WAS_TRIGGERED)):
                    self.env.robot.move_to_home(steps=60)
                    self.status_message = "ROBOT AT HOME"
                    self._update_3d_hud()

                self.env.step()

                if self.show_dashboard:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\n[Dashboard] Quit requested by user ('q' key).")
                        break
                    elif key in [ord('c'), ord('C')]:
                        self.set_camera_view()
                    elif key in [ord('1'), ord('2'), ord('3'), ord('4')]:
                        self.set_camera_view(int(chr(key)) - 1)

                time.sleep(0.005)

        except KeyboardInterrupt:
            print("\n\n[Playground] Shutting down...")
        finally:
            if self.show_dashboard:
                cv2.destroyAllWindows()
            self.env.close()
            print("[Playground] Simulation closed.")


def main():
    parser = argparse.ArgumentParser(description="CPS Interactive Simulation Playground (Deliverable 2)")
    parser.add_argument("--num-objects", type=int, default=3, help="Initial number of objects (1-8)")
    parser.add_argument("--no-dashboard", action="store_true", help="Disable OpenCV dashboard window")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/ggcnn_weights_cornell/ggcnn_epoch_23_cornell_statedict.pt",
                        help="Path to GG-CNN model weights")
    args = parser.parse_args()

    playground = InteractivePlayground(
        num_objects=args.num_objects,
        show_dashboard=not args.no_dashboard,
        checkpoint=args.checkpoint,
    )
    playground.run()


if __name__ == "__main__":
    main()
