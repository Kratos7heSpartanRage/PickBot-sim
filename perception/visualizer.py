"""
=============================================================================
  AI-Based Vision-Guided Robotic Grasping System: Deliverable 2
  Perception Subsystem: 4-Panel Industrial Vision Dashboard
=============================================================================
"""
import os
import cv2
import numpy as np


class Visualizer:
    """
    Visualization engine for Deliverable 2.
    Creates high-tech, publication-quality 4-panel figures and real-time OpenCV HUD overlays:
      Panel 1: RGB Perception & Predicted Grasps (with oriented antipodal jaws & target crosshairs)
      Panel 2: Metric Depth Field (Turbo Colormap)
      Panel 3: CNN Grasp Quality Map Q(u, v) (Inferno Colormap)
      Panel 4: Gripper Angle Map theta(u, v) (HSV Colormap)
    """
    def __init__(self):
        self.COLOR_TOP_GRASP = (0, 255, 60)      # Vivid Emerald Green
        self.COLOR_ALT_GRASP = (0, 215, 255)     # Warm Sunflower Yellow
        self.COLOR_CENTER = (0, 0, 255)          # Bright Red Center
        self.COLOR_CYAN = (255, 230, 0)          # Electric Cyan in BGR

    def draw_grasp_overlay(self, img, grasps, top_k=5):
        """
        Overlays grasp rectangles, jaw contacts, orientation indicators,
        and score labels onto an RGB image.
        """
        vis = img.copy()
        if len(vis.shape) == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

        for idx, grasp in enumerate(grasps[:top_k]):
            is_top = (idx == 0)
            color = self.COLOR_TOP_GRASP if is_top else self.COLOR_ALT_GRASP
            thickness = 3 if is_top else 2

            # 1. Draw grasp rectangle polygon
            pts = grasp.get_rect_points(jaw_length=18.0).astype(np.int32)
            cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=thickness)

            # 2. Highlight the two parallel gripper contact jaws
            left_jaw, right_jaw, _ = grasp.get_jaw_lines(jaw_length=18.0)
            p1_l, p2_l = tuple(np.int32(left_jaw[0])), tuple(np.int32(left_jaw[1]))
            p1_r, p2_r = tuple(np.int32(right_jaw[0])), tuple(np.int32(right_jaw[1]))
            cv2.line(vis, p1_l, p2_l, self.COLOR_CYAN, thickness + 1)
            cv2.line(vis, p1_r, p2_r, self.COLOR_CYAN, thickness + 1)

            # 3. Draw center crosshair
            c_x, c_y = int(round(grasp.u)), int(round(grasp.v))
            cv2.circle(vis, (c_x, c_y), 3, self.COLOR_CENTER, -1)

            # 4. Text badge on Top-1 grasp
            if is_top:
                badge = f"BEST: Q={grasp.quality:.2f} | {grasp.angle_deg:+.0f}deg"
                cv2.putText(vis, badge, (c_x - 36, c_y - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(vis, badge, (c_x - 36, c_y - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 100), 1, cv2.LINE_AA)
            else:
                badge = f"#{idx+1}: {grasp.quality:.2f}"
                cv2.putText(vis, badge, (c_x - 20, c_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        return vis

    def create_dashboard(self, rgb, depth, heatmaps, grasps, telemetry):
        """
        Constructs a 2x2 multi-panel industrial vision dashboard with sleek HUD banners:
        [Header HUD Banner: Live CPS Robotics Telemetry]
        [Panel 1: RGB + Grasp Overlays]   [Panel 2: Colorized Metric Depth]
        [Panel 3: Grasp Quality Map Q]    [Panel 4: Gripper Orientation Map theta]
        [Footer Command Bar: Quick Controls]
        """
        h, w = rgb.shape[:2]

        # Panel 1: RGB with Grasp Overlays
        panel1 = self.draw_grasp_overlay(rgb, grasps)

        # Panel 2: Depth map colorized (Turbo Colormap)
        d = depth.copy()
        d_norm = ((d - np.nanmin(d)) / (np.nanmax(d) - np.nanmin(d) + 1e-6) * 255).astype(np.uint8)
        panel2 = cv2.applyColorMap(d_norm, cv2.COLORMAP_TURBO)

        # Panel 3: Grasp Quality Heatmap Q(u, v) (Inferno)
        q_raw = heatmaps.get('q_map', np.zeros_like(depth))
        q_norm = (np.clip(q_raw, 0.0, 1.0) * 255).astype(np.uint8)
        panel3 = cv2.applyColorMap(q_norm, cv2.COLORMAP_INFERNO)
        for g in grasps[:5]:
            cv2.circle(panel3, (int(g.u), int(g.v)), 4, (0, 255, 255), 1)

        # Panel 4: Orientation Map theta(u, v) (HSV)
        ang = heatmaps.get('ang_map', np.zeros_like(depth))
        ang_norm = ((ang + np.pi / 2.0) / np.pi * 255.0).astype(np.uint8)
        panel4 = cv2.applyColorMap(ang_norm, cv2.COLORMAP_HSV)

        # Title cards on each of the 4 panels
        panel_titles = [
            "1. RGB Perception & Antipodal Grasps",
            "2. Metric Depth Field (Linear)",
            "3. GG-CNN Quality Heatmap Q(u, v)",
            "4. Gripper Angle Map theta(u, v)"
        ]
        for p_img, title in zip([panel1, panel2, panel3, panel4], panel_titles):
            cv2.rectangle(p_img, (0, 0), (w, 22), (18, 20, 24), -1)
            cv2.putText(p_img, title, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 235, 240), 1, cv2.LINE_AA)

        # Assemble 2x2 grid
        top_row = np.hstack([panel1, panel2])
        bottom_row = np.hstack([panel3, panel4])
        grid = np.vstack([top_row, bottom_row])

        # Top HUD Header (Sleek Dark Glass Style)
        hud_h = 46
        hud = np.full((hud_h, grid.shape[1], 3), 20, dtype=np.uint8)

        lat = telemetry.get('inference_time_ms', 0.0)
        fps = telemetry.get('fps', 0.0)
        dev = telemetry.get('device', 'cpu')
        top_q = telemetry.get('top_score', 0.0)

        t1 = "✦ CPS PICKBOT: AUTONOMOUS VISION-GUIDED MANIPULATION (GG-CNN) ✦"
        t2 = f"Inference: {lat:.1f} ms ({fps:.0f} FPS) | Hardware: {dev.upper()} | Best Quality: {top_q:.3f} | Targets: {len(grasps)}"
        if len(grasps) > 0 and grasps[0].world_coords is not None:
            wx, wy, wz = grasps[0].world_coords
            t2 += f" | 3D Pos: [{wx:.2f}, {wy:.2f}, {wz:.2f}]m | Angle: {grasps[0].angle_deg:+.1f}°"

        cv2.putText(hud, t1, (12, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 240, 210), 1, cv2.LINE_AA)
        cv2.putText(hud, t2, (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (215, 220, 225), 1, cv2.LINE_AA)

        # Bottom Command Bar
        bar_h = 24
        bar = np.full((bar_h, grid.shape[1], 3), 24, dtype=np.uint8)
        bar_text = "Controls: [1] Detect  |  [2] Next Pick  |  [3] Pick All (Clear)  |  [4] Pick Batch  |  [5] Respawn  |  [q] Quit"
        cv2.putText(bar, bar_text, (12, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 175, 190), 1, cv2.LINE_AA)

        dashboard = np.vstack([hud, grid, bar])
        return dashboard

    def save_report_figure(self, rgb, depth, heatmaps, grasps, telemetry, save_path="outputs/deliverable1_demo.png"):
        """Saves high-resolution publication-quality figure to disk."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        dashboard = self.create_dashboard(rgb, depth, heatmaps, grasps, telemetry)
        cv2.imwrite(save_path, cv2.cvtColor(dashboard, cv2.COLOR_RGB2BGR))
        print(f"Saved demonstration figure to: {save_path}")
        return save_path
