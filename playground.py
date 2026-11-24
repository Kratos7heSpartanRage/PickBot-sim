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

    def __init__(self, num_objects=3, show_dashboard=True, checkpoint=None, enable_gui_panels=False, font="dejavu", batch="all"):
        self.show_dashboard = show_dashboard
        self.num_objects = max(1, min(8, int(round(num_objects))))
        self.batch_count = 2
        self.batch = batch
        self.enable_gui_panels = enable_gui_panels
        self.gui_visible = enable_gui_panels
        self.console_window = "CPS Industrial Command Console"
        self._quit_requested = False
        self.font_family = font

        print("=" * 78)
        print("  ✦ CYBER PHYSICAL SYSTEMS: INTERACTIVE ROBOTIC PICK-AND-PLACE STUDIO ✦")
        print("  Autonomous AI-Based Vision-Guided Robotic Manipulation (Franka Panda 7-DOF)")
        print("=" * 78)
        print()

        # 1. Initialize PyBullet Simulation Environment (Clean Studio Mode by default)
        print("[1/3] Initializing PyBullet Simulation (Clean Studio Viewport)...")
        self.env = GraspingEnvironment(gui=True, enable_gui_panels=self.enable_gui_panels)

        # 2. Initialize Perception Model
        ckpt = checkpoint or "checkpoints/ggcnn_weights_cornell/ggcnn_epoch_23_cornell_statedict.pt"
        if not os.path.exists(ckpt):
            ckpt = "checkpoints/ggcnn_finetuned.pt"
        print(f"[2/3] Loading GG-CNN Perception Model: {ckpt}")
        self.detector = GraspDetector(checkpoint_path=ckpt)
        self.visualizer = Visualizer(font_family=self.font_family)
        self.executor = GraspExecutor(self.env.robot, tray_pos=(0.45, 0.45, 0.13))

        # 3. Spawn initial calibrated scene
        print(f"[3/3] Spawning {self.num_objects} calibrated objects ({self.batch.upper()}) in workspace...")
        self.env.reset(num_objects=self.num_objects, batch=self.batch)
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

        # Manual Teleoperation State (End-Effector Cartesian Position & Orientation)
        self.manual_arm_pos = [0.50, 0.00, 0.35]
        self.manual_arm_yaw = 0.0
        self.gripper_is_open = True

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

    def manual_jog_arm(self, dx=0.0, dy=0.0, dz=0.0, dyaw_deg=0.0, toggle_gripper=None):
        """Teleoperates the Franka Panda arm in Cartesian space via keyboard."""
        self.manual_arm_pos[0] = max(0.20, min(0.75, self.manual_arm_pos[0] + dx))
        self.manual_arm_pos[1] = max(-0.35, min(0.35, self.manual_arm_pos[1] + dy))
        self.manual_arm_pos[2] = max(0.03, min(0.60, self.manual_arm_pos[2] + dz))
        self.manual_arm_yaw += np.radians(dyaw_deg)

        if toggle_gripper is not None:
            if toggle_gripper:
                self.env.robot.open_gripper(steps=20)
                self.gripper_is_open = True
                print("  [Manual] Gripper Opened.")
            else:
                self.env.robot.close_gripper(force=120.0, steps=20)
                self.gripper_is_open = False
                print("  [Manual] Gripper Closed (120N squeeze).")

        self.env.robot.move_to_cartesian(
            self.manual_arm_pos,
            target_yaw=self.manual_arm_yaw,
            steps=10
        )
        self.status_message = f"MANUAL [{self.manual_arm_pos[0]:.2f}, {self.manual_arm_pos[1]:.2f}, {self.manual_arm_pos[2]:.2f}]"
        self._update_3d_hud()
        if self.show_dashboard:
            self.render_console()

    def _on_mouse_click(self, event, x, y, flags, param):
        """Dispatches mouse clicks on buttons and steppers in the OpenCV command console."""
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        action = self.visualizer.get_action_at(x, y)
        if not action:
            return

        print(f"\n[Console Click] Action: {action.upper()}")
        if action == "detect":
            self.run_detection()
        elif action == "execute":
            self.execute_single_grasp()
        elif action == "pick_all":
            self.batch_pick(target_count=None)
        elif action == "batch_pick":
            self.batch_pick(target_count=self.batch_count)
        elif action == "respawn":
            self.respawn_objects()
        elif action == "home":
            print("\n[Home] Returning robot arm smoothly to home posture...")
            self.env.robot.move_to_home(steps=60)
            self.manual_arm_pos = [0.50, 0.00, 0.35]
            self.manual_arm_yaw = 0.0
            self.status_message = "ROBOT AT HOME"
            self._update_3d_hud()
        elif action == "camera":
            self.set_camera_view()
        elif action == "quit":
            print("\n[Console] Quit requested via Quit button.")
            self._quit_requested = True
        elif action == "spawn_minus":
            self.num_objects = max(1, self.num_objects - 1)
            print(f"[Config] Spawn count: {self.num_objects}")
            self._update_3d_hud()
        elif action == "spawn_plus":
            self.num_objects = min(8, self.num_objects + 1)
            print(f"[Config] Spawn count: {self.num_objects}")
            self._update_3d_hud()
        elif action == "batch_minus":
            self.batch_count = max(1, self.batch_count - 1)
            print(f"[Config] Batch pick count: {self.batch_count}")
            self._update_3d_hud()
        elif action == "batch_plus":
            self.batch_count = min(8, self.batch_count + 1)
            print(f"[Config] Batch pick count: {self.batch_count}")
            self._update_3d_hud()

        self.render_console()

    def render_console(self):
        """Renders the 1440x820 Industrial Command Console and displays via OpenCV."""
        if not self.show_dashboard:
            return

        rgb, depth, _ = self.env.get_observation()
        clutter_n = len(self.get_table_objects())
        console = self.visualizer.create_command_console(
            rgb=rgb,
            depth=depth,
            heatmaps=self.last_heatmaps,
            grasps=self.last_grasps,
            telemetry=self.last_telemetry,
            force_snapshot=self.last_force_snapshot,
            status_msg=self.status_message,
            spawn_count=self.num_objects,
            batch_count=self.batch_count,
            success_stats=(self.successful_grasps, self.grasp_attempts),
            clutter_count=clutter_n
        )
        cv2.imshow(self.console_window, console)
        cv2.waitKey(1)

    def _setup_gui_controls(self):
        """Create PyBullet GUI sliders and action buttons (accessible when GUI panels are toggled on)."""
        # Section Header Text in 3D Scene
        p.addUserDebugText(
            "✦ CPS PICKBOT STUDIO ✦",
            [0.50, 0.00, 0.58],
            textColorRGB=[0.08, 0.88, 0.92],
            textSize=1.4
        )

        # Extra space padding + sacrificial marker so the first letters of all buttons & sliders are 100% visible
        PAD = "       >> "
        self.btn_detect = p.addUserDebugParameter(PAD + "[1] DETECT GRASPS <<", 1, 0, 0)
        self.btn_execute = p.addUserDebugParameter(PAD + "[2] EXECUTE NEXT GRASP <<", 1, 0, 0)
        self.btn_pick_all = p.addUserDebugParameter(PAD + "[3] PICK ALL (CLEAR SCENE) <<", 1, 0, 0)
        self.btn_batch_pick = p.addUserDebugParameter(PAD + "[4] PICK BATCH (COUNT) <<", 1, 0, 0)
        self.btn_respawn = p.addUserDebugParameter(PAD + "[5] RESPAWN OBJECTS <<", 1, 0, 0)
        self.btn_reset_robot = p.addUserDebugParameter(PAD + "[6] RESET ROBOT TO HOME <<", 1, 0, 0)
        self.btn_cam = p.addUserDebugParameter(PAD + "[7] SWITCH CAMERA VIEW <<", 1, 0, 0)

        # Configuration Sliders (Padded so 'Spawn' and 'Batch' first letters are completely intact)
        self.slider_num_obj = p.addUserDebugParameter(PAD + "Spawn Count [1 to 8]", 1, 8, self.num_objects)
        self.slider_batch_count = p.addUserDebugParameter(PAD + "Batch Pick Count [1 to 8]", 1, 8, self.batch_count)

        # Manual Arm Controls (Padded so 'Manual' first letter is completely intact)
        self.slider_x = p.addUserDebugParameter(PAD + "Manual Arm X", 0.20, 0.80, 0.50)
        self.slider_y = p.addUserDebugParameter(PAD + "Manual Arm Y", -0.40, 0.40, 0.00)
        self.slider_z = p.addUserDebugParameter(PAD + "Manual Arm Z", 0.05, 0.60, 0.35)
        self.slider_yaw = p.addUserDebugParameter(PAD + "Manual Gripper Yaw", -180, 180, 0)

        # Store previous button values to detect user clicks
        self.prev_detect = p.readUserDebugParameter(self.btn_detect)
        self.prev_execute = p.readUserDebugParameter(self.btn_execute)
        self.prev_pick_all = p.readUserDebugParameter(self.btn_pick_all)
        self.prev_batch_pick = p.readUserDebugParameter(self.btn_batch_pick)
        self.prev_respawn = p.readUserDebugParameter(self.btn_respawn)
        self.prev_reset_robot = p.readUserDebugParameter(self.btn_reset_robot)
        self.prev_cam = p.readUserDebugParameter(self.btn_cam)

    def toggle_gui(self):
        """Toggles PyBullet GUI sidebar panels on or off."""
        self.gui_visible = not self.gui_visible
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1 if self.gui_visible else 0)
        state = "PANELS ENABLED" if self.gui_visible else "CLEAN STUDIO MODE"
        print(f"\n[Visualizer] PyBullet Viewport: {state}")
        self.status_message = f"VIEW: {state}"
        self._update_3d_hud()

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
        print(f"\n[Camera Controller] Active View: {name}")
        self.status_message = f"VIEW: {name.split()[0]}"
        self._update_3d_hud()

    def _draw_tray_marker(self):
        """Draws a visual boundary box around the collection drop tray."""
        tx, ty, tz = 0.45, 0.45, 0.002
        half_w = 0.135
        p.addUserDebugLine([tx - half_w, ty - half_w, tz], [tx + half_w, ty - half_w, tz], [0.95, 0.78, 0.10], 2.0)
        p.addUserDebugLine([tx + half_w, ty - half_w, tz], [tx + half_w, ty + half_w, tz], [0.95, 0.78, 0.10], 2.0)
        p.addUserDebugLine([tx + half_w, ty + half_w, tz], [tx - half_w, ty + half_w, tz], [0.95, 0.78, 0.10], 2.0)
        p.addUserDebugLine([tx - half_w, ty + half_w, tz], [tx - half_w, ty - half_w, tz], [0.95, 0.78, 0.10], 2.0)
        # Position label on open table surface in front of yellow border to prevent intersecting the tray mesh
        p.addUserDebugText("COLLECTION TRAY", [tx - 0.08, ty - 0.20, tz + 0.005], [0.95, 0.78, 0.10], 0.85)

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
        spawn_n = self.num_objects
        batch_n = self.batch_count

        rate_str = f"{100.0 * self.successful_grasps / max(1, self.grasp_attempts):.0f}%" if self.grasp_attempts > 0 else "N/A"

        line1 = f"Status: [{self.status_message}] | Table: {len(table_objs)} objs | Tray: {len(tray_objs)} objs"
        line2 = f"Success: {rate_str} ({self.successful_grasps}/{self.grasp_attempts}) | Config: Spawn [{spawn_n}] | Batch [{batch_n}] | Set: [{self.batch.upper()}]"

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
        print("=" * 78)
        print("  ✦ CYBER-PHYSICAL PICKBOT STUDIO: CLEAN COMMAND & HOTKEY REFERENCE ✦")
        print("=" * 78)
        print("  Controls work seamlessly from EITHER the PyBullet 3D Window OR OpenCV Dashboard:")
        print()
        print("  [Autonomous & Manipulation Actions]")
        print("    'd' or 'D'         : Run GG-CNN Grasp Detection & Visualizer")
        print("    'e' or [Space]     : Execute Next Best Grasp (Pick -> Lift -> Tray)")
        print("    'a' or 'A'         : Pick All (Autonomous loop until table is fully cleared)")
        print("    'b' or 'B'         : Pick Batch (Picks exact count set by Batch slider)")
        print("    'r' or 'R'         : Respawn Scene (Clears & spawns randomized objects)")
        print("    'h' or 'H'         : Reset Robot Home (Smooth return to neutral posture)")
        print()
        print("  [Manual Gripper Teleoperation & Jogging]")
        print("    Arrow Keys (↑ / ↓) : Move Gripper Forward / Backward along table (X axis)")
        print("    Arrow Keys (← / →) : Move Gripper Left / Right along table (Y axis)")
        print("    PageUp / PageDown  : Move Gripper Up / Down (Z axis) [or 'u' / 'j']")
        print("    '<' or '>' (,/.)   : Rotate Gripper Yaw Angle (Wrist orientation)")
        print("    'o' or 'O'         : Open Gripper Fingers (to 8cm width)")
        print("    'p' or 'P'         : Close / Squeeze Gripper (120N force closure)")
        print()
        print("  [3D Camera Presets]")
        print("    '1'                : Isometric Studio (3D Global Overview)")
        print("    '2'                : Over-The-Shoulder (OTS Workspace Alignment)")
        print("    '3'                : Top-Down Overhead (90° Metric Inspection)")
        print("    '4'                : Side Clearance Profile (Lift & Height Verification)")
        print("    'c' or 'C'         : Cycle Through Camera Presets (1 -> 2 -> 3 -> 4)")
        print()
        print("  [Configuration & Telemetry]")
        print("    '+' / '-'          : Step Spawn Count (+1 / -1)")
        print("    ']' / '['          : Step Batch Pick Count (+1 / -1)")
        print("    't' or 'T'         : Cycle Object Catalog Batch (all -> batch1 -> batch2)")
        print("    'g' or 'G'         : Toggle PyBullet Sidebars (Clean Studio <-> Debug Panels)")
        print("    'q' or [Esc]       : Quit Simulation")
        print()
        print("  [Live Telemetry Displays]")
        print("    - Floating 3D HUD  : Real-time Status, Clutter Counts, Force Closure & Joint Torques")
        print("    - OpenCV Window    : 4-Panel RGB-D, Inferno Heatmap, Angle Quivers & Trackbars")
        print("=" * 78)
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

        # Update OpenCV Command Console
        if self.show_dashboard:
            self.render_console()

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
            self.render_console()

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
            self.render_console()

    def respawn_objects(self, count=None, batch=None):
        """Clear scene and spawn randomized objects matching the integer count and active batch."""
        if batch is not None:
            self.batch = batch
        num = count if count is not None else self.num_objects
        num = max(1, min(8, int(round(num))))
        self.num_objects = num
        print(f"\n[Respawn] Spawning {num} new calibrated objects from set [{self.batch.upper()}]...")
        self._clear_grasp_markers()
        self.last_grasps = []
        self.env.reset(num_objects=num, batch=self.batch)
        for _ in range(60):
            self.env.step()
        self.status_message = f"RESPAWNED {num} OBJECTS [{self.batch.upper()}]"
        self._update_3d_hud()
        self.render_console()
        obj_info = self.env.spawner.get_object_info()
        print(f"  Spawned: {', '.join(o['name'] for o in obj_info)}")

    def cycle_batch(self):
        """Cycles object library batch: all -> batch1 -> batch2 -> all, then respawns."""
        batch_order = ["all", "batch1", "batch2"]
        curr_idx = batch_order.index(self.batch) if self.batch in batch_order else 0
        self.batch = batch_order[(curr_idx + 1) % len(batch_order)]
        label_map = {
            "all": "Full 16-Object Catalog (Batch 1 + Batch 2)",
            "batch1": "Batch 1 Only (Primary Baseline 8 Objects)",
            "batch2": "Batch 2 Only (Novel Geometric Variation 8 Objects)"
        }
        print(f"\n[Catalog] Switched active batch to: {label_map[self.batch]}")
        self.respawn_objects()

    def run(self):
        """Main interactive loop."""
        mode_str = "Legacy GUI Panels Visible" if self.gui_visible else "Clean Studio Viewport (0 Debug Panels)"
        print(f"\n[Playground] Active! {mode_str}.\n"
              f"  Interactive surface: {self.console_window} (1440x820)\n"
              f"  Click buttons/steppers in the console, or use keyboard shortcuts in either window.\n"
              f"  Press 'g' to toggle PyBullet sidebars, or 'q' / Esc to exit.\n")

        if self.show_dashboard:
            cv2.namedWindow(self.console_window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.console_window, 1440, 820)
            cv2.setMouseCallback(self.console_window, self._on_mouse_click)
            self.render_console()

        # Track previous slider values to only move robot on explicit slider adjustment
        prev_slider_vals = (
            p.readUserDebugParameter(self.slider_x),
            p.readUserDebugParameter(self.slider_y),
            p.readUserDebugParameter(self.slider_z),
            p.readUserDebugParameter(self.slider_yaw)
        )
        prev_spawn_n = self.num_objects
        prev_batch_n = self.batch_count
        last_console_time = 0.0

        try:
            while not self._quit_requested:
                now = time.time()

                # 1. Check if PyBullet configuration sliders changed (when GUI panels are enabled)
                if self.gui_visible:
                    curr_spawn_n = int(round(p.readUserDebugParameter(self.slider_num_obj)))
                    curr_batch_n = int(round(p.readUserDebugParameter(self.slider_batch_count)))
                    if curr_spawn_n != prev_spawn_n or curr_batch_n != prev_batch_n:
                        prev_spawn_n = curr_spawn_n
                        prev_batch_n = curr_batch_n
                        self.num_objects = curr_spawn_n
                        self.batch_count = curr_batch_n
                        self._update_3d_hud()
                        self.render_console()

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
                    curr_detect = p.readUserDebugParameter(self.btn_detect)
                    if curr_detect != self.prev_detect:
                        self.prev_detect = curr_detect
                        self.run_detection()

                    curr_execute = p.readUserDebugParameter(self.btn_execute)
                    if curr_execute != self.prev_execute:
                        self.prev_execute = curr_execute
                        self.execute_single_grasp()

                    curr_pick_all = p.readUserDebugParameter(self.btn_pick_all)
                    if curr_pick_all != self.prev_pick_all:
                        self.prev_pick_all = curr_pick_all
                        self.batch_pick(target_count=None)

                    curr_batch_pick = p.readUserDebugParameter(self.btn_batch_pick)
                    if curr_batch_pick != self.prev_batch_pick:
                        self.prev_batch_pick = curr_batch_pick
                        self.batch_pick(target_count=self.batch_count)

                    curr_respawn = p.readUserDebugParameter(self.btn_respawn)
                    if curr_respawn != self.prev_respawn:
                        self.prev_respawn = curr_respawn
                        self.respawn_objects()

                    curr_reset = p.readUserDebugParameter(self.btn_reset_robot)
                    if curr_reset != self.prev_reset_robot:
                        self.prev_reset_robot = curr_reset
                        print("\n[Reset] Returning robot arm smoothly to home posture...")
                        self.env.robot.move_to_home(steps=60)
                        self.status_message = "ROBOT AT HOME"
                        self._update_3d_hud()
                        self.render_console()

                    curr_cam = p.readUserDebugParameter(self.btn_cam)
                    if curr_cam != self.prev_cam:
                        self.prev_cam = curr_cam
                        self.set_camera_view()

                # 4. Check Keyboard Shortcuts in PyBullet GUI Window
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
                   (ord('E') in keys and (keys[ord('E')] & p.KEY_WAS_TRIGGERED)) or \
                   (32 in keys and (keys[32] & p.KEY_WAS_TRIGGERED)):
                    self.execute_single_grasp()
                if (ord('a') in keys and (keys[ord('a')] & p.KEY_WAS_TRIGGERED)) or \
                   (ord('A') in keys and (keys[ord('A')] & p.KEY_WAS_TRIGGERED)):
                    self.batch_pick(target_count=None)
                if (ord('b') in keys and (keys[ord('b')] & p.KEY_WAS_TRIGGERED)) or \
                   (ord('B') in keys and (keys[ord('B')] & p.KEY_WAS_TRIGGERED)):
                    self.batch_pick(target_count=self.batch_count)
                if (ord('r') in keys and (keys[ord('r')] & p.KEY_WAS_TRIGGERED)) or \
                   (ord('R') in keys and (keys[ord('R')] & p.KEY_WAS_TRIGGERED)):
                    self.respawn_objects()
                if (ord('h') in keys and (keys[ord('h')] & p.KEY_WAS_TRIGGERED)) or \
                   (ord('H') in keys and (keys[ord('H')] & p.KEY_WAS_TRIGGERED)):
                    self.env.robot.move_to_home(steps=60)
                    self.manual_arm_pos = [0.50, 0.00, 0.35]
                    self.manual_arm_yaw = 0.0
                    self.status_message = "ROBOT AT HOME"
                    self._update_3d_hud()
                    self.render_console()

                # Manual Jogging Controls (PyBullet)
                if p.B3G_UP_ARROW in keys and (keys[p.B3G_UP_ARROW] & (p.KEY_WAS_TRIGGERED | p.KEY_IS_DOWN)):
                    self.manual_jog_arm(dx=+0.015)
                if p.B3G_DOWN_ARROW in keys and (keys[p.B3G_DOWN_ARROW] & (p.KEY_WAS_TRIGGERED | p.KEY_IS_DOWN)):
                    self.manual_jog_arm(dx=-0.015)
                if p.B3G_LEFT_ARROW in keys and (keys[p.B3G_LEFT_ARROW] & (p.KEY_WAS_TRIGGERED | p.KEY_IS_DOWN)):
                    self.manual_jog_arm(dy=+0.015)
                if p.B3G_RIGHT_ARROW in keys and (keys[p.B3G_RIGHT_ARROW] & (p.KEY_WAS_TRIGGERED | p.KEY_IS_DOWN)):
                    self.manual_jog_arm(dy=-0.015)
                if (p.B3G_PAGE_UP in keys and (keys[p.B3G_PAGE_UP] & (p.KEY_WAS_TRIGGERED | p.KEY_IS_DOWN))) or \
                   (ord('u') in keys and (keys[ord('u')] & p.KEY_WAS_TRIGGERED)) or \
                   (ord('U') in keys and (keys[ord('U')] & p.KEY_WAS_TRIGGERED)):
                    self.manual_jog_arm(dz=+0.015)
                if (p.B3G_PAGE_DOWN in keys and (keys[p.B3G_PAGE_DOWN] & (p.KEY_WAS_TRIGGERED | p.KEY_IS_DOWN))) or \
                   (ord('j') in keys and (keys[ord('j')] & p.KEY_WAS_TRIGGERED)) or \
                   (ord('J') in keys and (keys[ord('J')] & p.KEY_WAS_TRIGGERED)):
                    self.manual_jog_arm(dz=-0.015)
                if (ord(',') in keys and (keys[ord(',')] & p.KEY_WAS_TRIGGERED)) or \
                   (ord('<') in keys and (keys[ord('<')] & p.KEY_WAS_TRIGGERED)):
                    self.manual_jog_arm(dyaw_deg=+10.0)
                if (ord('.') in keys and (keys[ord('.')] & p.KEY_WAS_TRIGGERED)) or \
                   (ord('>') in keys and (keys[ord('>')] & p.KEY_WAS_TRIGGERED)):
                    self.manual_jog_arm(dyaw_deg=-10.0)
                if (ord('o') in keys and (keys[ord('o')] & p.KEY_WAS_TRIGGERED)) or \
                   (ord('O') in keys and (keys[ord('O')] & p.KEY_WAS_TRIGGERED)):
                    self.manual_jog_arm(toggle_gripper=True)
                if (ord('p') in keys and (keys[ord('p')] & p.KEY_WAS_TRIGGERED)) or \
                   (ord('P') in keys and (keys[ord('P')] & p.KEY_WAS_TRIGGERED)):
                    self.manual_jog_arm(toggle_gripper=False)
                if (ord('g') in keys and (keys[ord('g')] & p.KEY_WAS_TRIGGERED)) or \
                   (ord('G') in keys and (keys[ord('G')] & p.KEY_WAS_TRIGGERED)):
                    self.toggle_gui()
                if (ord('+') in keys and (keys[ord('+')] & p.KEY_WAS_TRIGGERED)) or \
                   (ord('=') in keys and (keys[ord('=')] & p.KEY_WAS_TRIGGERED)):
                    self.num_objects = min(8, self.num_objects + 1)
                    self._update_3d_hud()
                    self.render_console()
                if (ord('-') in keys and (keys[ord('-')] & p.KEY_WAS_TRIGGERED)) or \
                   (ord('_') in keys and (keys[ord('_')] & p.KEY_WAS_TRIGGERED)):
                    self.num_objects = max(1, self.num_objects - 1)
                    self._update_3d_hud()
                    self.render_console()
                if ord(']') in keys and (keys[ord(']')] & p.KEY_WAS_TRIGGERED):
                    self.batch_count = min(8, self.batch_count + 1)
                    self._update_3d_hud()
                    self.render_console()
                if ord('[') in keys and (keys[ord('[')] & p.KEY_WAS_TRIGGERED):
                    self.batch_count = max(1, self.batch_count - 1)
                    self._update_3d_hud()
                    self.render_console()
                if (ord('t') in keys and (keys[ord('t')] & p.KEY_WAS_TRIGGERED)) or \
                   (ord('T') in keys and (keys[ord('T')] & p.KEY_WAS_TRIGGERED)):
                    self.cycle_batch()
                if (ord('q') in keys and (keys[ord('q')] & p.KEY_WAS_TRIGGERED)) or \
                   (ord('Q') in keys and (keys[ord('Q')] & p.KEY_WAS_TRIGGERED)):
                    print("\n[PyBullet] Quit requested by user ('q' key).")
                    break

                self.env.step()

                # 5. Check Keyboard Shortcuts in OpenCV Command Console
                if self.show_dashboard:
                    raw_k = cv2.waitKey(1)
                    if raw_k != -1:
                        key = raw_k & 0xFF
                        if key in [ord('q'), ord('Q'), 27]:
                            print("\n[Console] Quit requested by user ('q' key).")
                            break
                        elif key in [ord('c'), ord('C')]:
                            self.set_camera_view()
                        elif key in [ord('1'), ord('2'), ord('3'), ord('4')]:
                            self.set_camera_view(int(chr(key)) - 1)
                        elif key in [ord('d'), ord('D')]:
                            self.run_detection()
                        elif key in [ord('e'), ord('E'), ord(' ')]:
                            self.execute_single_grasp()
                        elif key in [ord('a'), ord('A')]:
                            self.batch_pick(target_count=None)
                        elif key in [ord('b'), ord('B')]:
                            self.batch_pick(target_count=self.batch_count)
                        elif key in [ord('r'), ord('R')]:
                            self.respawn_objects()
                        elif key in [ord('h'), ord('H')]:
                            self.env.robot.move_to_home(steps=60)
                            self.manual_arm_pos = [0.50, 0.00, 0.35]
                            self.manual_arm_yaw = 0.0
                            self.status_message = "ROBOT AT HOME"
                            self._update_3d_hud()
                            self.render_console()
                        elif key in [ord('g'), ord('G')]:
                            self.toggle_gui()
                        elif key in [ord('u'), ord('U')]:
                            self.manual_jog_arm(dz=+0.02)
                        elif key in [ord('j'), ord('J')]:
                            self.manual_jog_arm(dz=-0.02)
                        elif key in [ord(','), ord('<')]:
                            self.manual_jog_arm(dyaw_deg=+10.0)
                        elif key in [ord('.'), ord('>')]:
                            self.manual_jog_arm(dyaw_deg=-10.0)
                        elif key in [ord('o'), ord('O')]:
                            self.manual_jog_arm(toggle_gripper=True)
                        elif key in [ord('p'), ord('P')]:
                            self.manual_jog_arm(toggle_gripper=False)
                        elif key in [ord('+'), ord('=')]:
                            self.num_objects = min(8, self.num_objects + 1)
                            self._update_3d_hud()
                            self.render_console()
                        elif key in [ord('-'), ord('_')]:
                            self.num_objects = max(1, self.num_objects - 1)
                            self._update_3d_hud()
                            self.render_console()
                        elif key == ord(']'):
                            self.batch_count = min(8, self.batch_count + 1)
                            self._update_3d_hud()
                            self.render_console()
                        elif key == ord('['):
                            self.batch_count = max(1, self.batch_count - 1)
                            self._update_3d_hud()
                            self.render_console()
                        elif key in [ord('t'), ord('T')]:
                            self.cycle_batch()

                # Refresh console at ~25-30 FPS to keep telemetry live
                if self.show_dashboard and (now - last_console_time > 0.035):
                    self.render_console()
                    last_console_time = now

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
    parser.add_argument("--no-dashboard", action="store_true", help="Disable OpenCV command console window")
    parser.add_argument("--show-gui-panels", action="store_true", help="Start with legacy PyBullet sidebar GUI panels visible (Clean Studio Mode is default)")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/ggcnn_weights_cornell/ggcnn_epoch_23_cornell_statedict.pt",
                        help="Path to GG-CNN model weights")
    parser.add_argument("--font", type=str, default="dejavu",
                        help="Console font preset ('dejavu', 'mono', 'ubuntu', 'ubuntu_mono') or path to custom .ttf/.otf")
    parser.add_argument("--batch", type=str, default="all", choices=["all", "batch1", "batch2"],
                        help="Object catalog batch: 'all' (all 16 objects), 'batch1' (objects 1-8), or 'batch2' (objects 9-16 novel geometric set)")
    args = parser.parse_args()

    playground = InteractivePlayground(
        num_objects=args.num_objects,
        show_dashboard=not args.no_dashboard,
        enable_gui_panels=args.show_gui_panels,
        checkpoint=args.checkpoint,
        font=args.font,
        batch=args.batch,
    )
    playground.run()


if __name__ == "__main__":
    main()
