#!/usr/bin/env python3
"""
=================================================================
  CYBER PHYSICAL SYSTEMS: INTERACTIVE SIMULATION PLAYGROUND
=================================================================
An interactive simulation environment for exploring vision-guided
robotic grasping. Features:
  - PyBullet 3D GUI with interactive robot control sliders
  - Real-time CNN grasp detection with visual feedback
  - Live OpenCV perception dashboard
  - Object spawning, respawning, and scene manipulation
  - Grasp execution: detect, approach, grasp, and lift objects

Usage:
    python playground.py
    python playground.py --num-objects 5
    python playground.py --no-dashboard  (PyBullet only, no OpenCV window)
=================================================================
"""
import argparse
import pybullet as p
import time
import sys
import os
import numpy as np
import cv2

from simulation.environment import GraspingEnvironment
from perception.detector import GraspDetector
from perception.visualizer import Visualizer


class InteractivePlayground:
    """Full interactive playground with PyBullet GUI + OpenCV dashboard."""

    def __init__(self, num_objects=3, show_dashboard=True, checkpoint=None):
        self.show_dashboard = show_dashboard
        self.num_objects = num_objects

        print("=" * 70)
        print("  CYBER PHYSICAL SYSTEMS: INTERACTIVE PLAYGROUND")
        print("  AI-Based Vision-Guided Robotic Grasping System")
        print("=" * 70)
        print()

        # 1. Initialize PyBullet Environment
        print("[1/3] Initializing PyBullet Simulation with GUI...")
        self.env = GraspingEnvironment(gui=True)

        # 2. Initialize Perception
        ckpt = checkpoint or "checkpoints/ggcnn_weights_cornell/ggcnn_epoch_23_cornell_statedict.pt"
        print(f"[2/3] Loading GG-CNN Perception Model: {ckpt}")
        self.detector = GraspDetector(checkpoint_path=ckpt)
        self.visualizer = Visualizer()

        # 3. Spawn initial scene
        print(f"[3/3] Spawning {num_objects} objects in workspace...")
        self.env.reset(num_objects=num_objects)
        for _ in range(60):
            self.env.step()

        # State tracking
        self.last_grasps = []
        self.last_telemetry = {}
        self.last_heatmaps = {}
        self.grasp_markers = []
        self.detection_count = 0
        self.grasp_attempts = 0
        self.successful_grasps = 0

        # Setup interactive GUI controls
        self._setup_gui_controls()

        # Print help
        self._print_help()

    def _setup_gui_controls(self):
        """Create PyBullet GUI sliders and buttons."""
        # Title text
        p.addUserDebugText(
            "CPS Interactive Playground",
            [0.5, 0.0, 0.55],
            textColorRGB=[1.0, 0.85, 0.0],
            textSize=1.5
        )
        p.addUserDebugText(
            "Use sliders on the right panel -->",
            [0.5, 0.0, 0.50],
            textColorRGB=[0.7, 0.7, 0.7],
            textSize=1.0
        )

        # Robot end-effector control sliders
        self.slider_x = p.addUserDebugParameter("Robot X", 0.2, 0.8, 0.5)
        self.slider_y = p.addUserDebugParameter("Robot Y", -0.4, 0.4, 0.0)
        self.slider_z = p.addUserDebugParameter("Robot Z", 0.0, 0.6, 0.35)
        self.slider_yaw = p.addUserDebugParameter("Gripper Yaw (deg)", -180, 180, 0)

        # Object count slider
        self.slider_num_obj = p.addUserDebugParameter("Num Objects (respawn)", 1, 8, self.num_objects)

        # Action buttons (PyBullet buttons are sliders that increment on click)
        self.btn_detect = p.addUserDebugParameter(">> DETECT GRASPS <<", 1, 0, 0)
        self.btn_execute = p.addUserDebugParameter(">> EXECUTE TOP GRASP <<", 1, 0, 0)
        self.btn_respawn = p.addUserDebugParameter(">> RESPAWN OBJECTS <<", 1, 0, 0)
        self.btn_reset_robot = p.addUserDebugParameter(">> RESET ROBOT <<", 1, 0, 0)

        # Store previous button values to detect clicks
        self.prev_detect = p.readUserDebugParameter(self.btn_detect)
        self.prev_execute = p.readUserDebugParameter(self.btn_execute)
        self.prev_respawn = p.readUserDebugParameter(self.btn_respawn)
        self.prev_reset_robot = p.readUserDebugParameter(self.btn_reset_robot)

    def _print_help(self):
        """Print interactive controls help."""
        print()
        print("=" * 70)
        print("  CONTROLS:")
        print("=" * 70)
        print("  [PyBullet GUI Panel - Right Side]")
        print("    Robot X/Y/Z     : Move the robot arm end-effector")
        print("    Gripper Yaw     : Rotate the gripper orientation")
        print("    Num Objects     : Set count for next respawn")
        print()
        print("  [Action Buttons]")
        print("    DETECT GRASPS   : Run CNN inference on current scene")
        print("    EXECUTE GRASP   : Move robot to top grasp, close gripper, lift")
        print("    RESPAWN OBJECTS  : Clear and spawn new random objects")
        print("    RESET ROBOT     : Return robot to home position")
        print()
        print("  [OpenCV Dashboard Window]")
        print("    Shows live RGB, depth, quality heatmap, and angle map")
        print("    Updated each time you run DETECT GRASPS")
        print()
        print("  Press Ctrl+C in terminal to exit.")
        print("=" * 70)
        print()

    def _clear_grasp_markers(self):
        """Remove all visual debug markers from PyBullet."""
        for marker_id in self.grasp_markers:
            try:
                p.removeUserDebugItem(marker_id)
            except Exception:
                pass
        self.grasp_markers.clear()

    def _draw_grasp_in_sim(self, grasp, rank=0):
        """Draw a grasp visualization in the 3D PyBullet scene."""
        if grasp.world_coords is None:
            return

        wx, wy, wz = grasp.world_coords
        is_top = (rank == 0)

        # Colors: green for top grasp, yellow for alternatives
        color = [0.0, 1.0, 0.2] if is_top else [1.0, 0.85, 0.0]
        width = 4.0 if is_top else 2.0

        # Draw approach line (vertical above grasp point)
        m1 = p.addUserDebugLine(
            [wx, wy, wz], [wx, wy, wz + 0.15],
            lineColorRGB=color, lineWidth=width, lifeTime=0
        )
        self.grasp_markers.append(m1)

        # Draw small crosshair at grasp point
        offset = 0.02
        m2 = p.addUserDebugLine(
            [wx - offset, wy, wz], [wx + offset, wy, wz],
            lineColorRGB=color, lineWidth=width, lifeTime=0
        )
        m3 = p.addUserDebugLine(
            [wx, wy - offset, wz], [wx, wy + offset, wz],
            lineColorRGB=color, lineWidth=width, lifeTime=0
        )
        self.grasp_markers.extend([m2, m3])

        # Draw gripper opening direction based on angle
        angle = grasp.angle_rad
        dx = np.cos(angle) * 0.03
        dy = np.sin(angle) * 0.03
        m4 = p.addUserDebugLine(
            [wx - dx, wy - dy, wz], [wx + dx, wy + dy, wz],
            lineColorRGB=[0.0, 0.7, 1.0], lineWidth=3.0, lifeTime=0
        )
        self.grasp_markers.append(m4)

        # Text label for top grasp
        if is_top:
            m5 = p.addUserDebugText(
                f"Q={grasp.quality:.2f} | {grasp.angle_deg:+.0f}deg",
                [wx, wy, wz + 0.17],
                textColorRGB=[1.0, 1.0, 1.0],
                textSize=1.2
            )
            self.grasp_markers.append(m5)

    def run_detection(self):
        """Run CNN grasp detection on the current scene."""
        self.detection_count += 1
        self._clear_grasp_markers()

        print(f"\n[Detection #{self.detection_count}] Running GG-CNN inference...")

        rgb, depth, seg = self.env.get_observation()
        grasps, telemetry, heatmaps = self.detector.detect(
            rgb, depth, camera=self.env.camera, top_k=5
        )

        self.last_grasps = grasps
        self.last_telemetry = telemetry
        self.last_heatmaps = heatmaps

        # Print results
        print(f"  Latency   : {telemetry['inference_time_ms']:.2f} ms")
        print(f"  Throughput: {telemetry['fps']:.0f} FPS")
        print(f"  Device    : {telemetry['device'].upper()}")
        print(f"  Detected  : {len(grasps)} grasp candidates")

        # Show objects in scene
        obj_info = self.env.spawner.get_object_info()
        print(f"  Objects   : {', '.join(o['name'] for o in obj_info)}")

        if grasps:
            top = grasps[0]
            print(f"  Top Grasp : Q={top.quality:.3f}, angle={top.angle_deg:+.1f}deg", end="")
            if top.world_coords is not None:
                wx, wy, wz = top.world_coords
                print(f", pos=[{wx:.3f}, {wy:.3f}, {wz:.3f}]")
            else:
                print()

        # Draw grasps in 3D simulation
        for rank, g in enumerate(grasps[:5]):
            self._draw_grasp_in_sim(g, rank=rank)

        # Update OpenCV dashboard
        if self.show_dashboard:
            dashboard = self.visualizer.create_dashboard(rgb, depth, heatmaps, grasps, telemetry)
            cv2.imshow("CPS Grasp Detection Dashboard", cv2.cvtColor(dashboard, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

    def execute_grasp(self):
        """Execute a full pick sequence: approach, descend, grasp, lift."""
        if not self.last_grasps:
            print("\n[Execute] No grasps detected yet! Run DETECT GRASPS first.")
            return

        top = self.last_grasps[0]
        if top.world_coords is None:
            print("\n[Execute] Top grasp has no 3D coordinates!")
            return

        self.grasp_attempts += 1
        wx, wy, wz = top.world_coords
        yaw = top.angle_rad

        print(f"\n[Execute Grasp #{self.grasp_attempts}]")
        print(f"  Target: [{wx:.3f}, {wy:.3f}, {wz:.3f}], yaw={top.angle_deg:+.1f}deg")

        # Phase 1: Approach (move above target)
        approach_z = wz + 0.15
        print(f"  Phase 1: Approach -> z={approach_z:.3f}m")
        self.env.robot.move_to_cartesian([wx, wy, approach_z], target_yaw=yaw, steps=120)
        for _ in range(30):
            self.env.step()

        # Phase 2: Descend to grasp height
        grasp_z = wz + 0.02
        print(f"  Phase 2: Descend -> z={grasp_z:.3f}m")
        self.env.robot.move_to_cartesian([wx, wy, grasp_z], target_yaw=yaw, steps=100)
        for _ in range(30):
            self.env.step()

        # Phase 3: Close gripper
        print("  Phase 3: Closing gripper...")
        for g_idx in self.env.robot.gripper_joints:
            p.setJointMotorControl2(
                self.env.robot.robot_id, g_idx,
                p.POSITION_CONTROL, targetPosition=0.0, force=40.0
            )
        for _ in range(60):
            self.env.step()

        # Phase 4: Lift
        lift_z = wz + 0.25
        print(f"  Phase 4: Lift -> z={lift_z:.3f}m")
        self.env.robot.move_to_cartesian([wx, wy, lift_z], target_yaw=yaw, steps=120)
        for _ in range(60):
            self.env.step()

        # Check if object was grasped (look if any object moved significantly upward)
        grasped = False
        for data in self.env.spawner.object_data:
            try:
                pos, _ = p.getBasePositionAndOrientation(data["id"])
                if pos[2] > 0.10:
                    grasped = True
                    break
            except Exception:
                pass

        if grasped:
            self.successful_grasps += 1
            print(f"  Result: SUCCESS! Object lifted.")
            # Move to tray and drop
            print("  Phase 5: Moving to tray and releasing...")
            self.env.robot.move_to_cartesian([0.5, 0.35, 0.3], target_yaw=0, steps=120)
            for _ in range(30):
                self.env.step()
            # Open gripper
            for g_idx in self.env.robot.gripper_joints:
                p.setJointMotorControl2(
                    self.env.robot.robot_id, g_idx,
                    p.POSITION_CONTROL, targetPosition=0.04, force=20.0
                )
            for _ in range(60):
                self.env.step()
        else:
            print(f"  Result: MISS - object not grasped.")

        print(f"  Stats: {self.successful_grasps}/{self.grasp_attempts} successful grasps "
              f"({100*self.successful_grasps/max(1,self.grasp_attempts):.0f}%)")

        # Return robot home
        print("  Returning robot to home position...")
        self.env.robot.reset()

    def respawn_objects(self):
        """Clear and respawn objects with the current slider count."""
        num = int(round(p.readUserDebugParameter(self.slider_num_obj)))
        num = max(1, min(num, 8))
        print(f"\n[Respawn] Spawning {num} new random objects...")
        self._clear_grasp_markers()
        self.last_grasps = []
        self.env.reset(num_objects=num)
        for _ in range(60):
            self.env.step()
        obj_info = self.env.spawner.get_object_info()
        print(f"  Spawned: {', '.join(o['name'] for o in obj_info)}")

    def run(self):
        """Main interactive loop."""
        print("\n[Playground] Running... Use GUI controls or press Ctrl+C to exit.\n")

        if self.show_dashboard:
            # Create initial empty dashboard
            cv2.namedWindow("CPS Grasp Detection Dashboard", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("CPS Grasp Detection Dashboard", 800, 500)

        frame_count = 0
        try:
            while True:
                # Read robot control sliders and move arm
                tx = p.readUserDebugParameter(self.slider_x)
                ty = p.readUserDebugParameter(self.slider_y)
                tz = p.readUserDebugParameter(self.slider_z)
                tyaw_deg = p.readUserDebugParameter(self.slider_yaw)
                tyaw = np.radians(tyaw_deg)

                self.env.robot.move_to_cartesian([tx, ty, tz], target_yaw=tyaw, steps=1)

                # Check button presses
                curr_detect = p.readUserDebugParameter(self.btn_detect)
                if curr_detect != self.prev_detect:
                    self.prev_detect = curr_detect
                    self.run_detection()

                curr_execute = p.readUserDebugParameter(self.btn_execute)
                if curr_execute != self.prev_execute:
                    self.prev_execute = curr_execute
                    self.execute_grasp()

                curr_respawn = p.readUserDebugParameter(self.btn_respawn)
                if curr_respawn != self.prev_respawn:
                    self.prev_respawn = curr_respawn
                    self.respawn_objects()

                curr_reset = p.readUserDebugParameter(self.btn_reset_robot)
                if curr_reset != self.prev_reset_robot:
                    self.prev_reset_robot = curr_reset
                    print("\n[Reset] Returning robot to home position...")
                    self.env.robot.reset()

                self.env.step()

                if self.show_dashboard:
                    cv2.waitKey(1)

                frame_count += 1
                time.sleep(0.005)

        except KeyboardInterrupt:
            print("\n\n[Playground] Shutting down...")
        finally:
            if self.show_dashboard:
                cv2.destroyAllWindows()
            self.env.close()
            print("[Playground] Goodbye!")


def main():
    parser = argparse.ArgumentParser(description="CPS Interactive Simulation Playground")
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
