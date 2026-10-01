import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

class Visualizer:
    """
    Visualization engine for Deliverable 1.
    Creates publication-quality 4-panel figures and real-time OpenCV HUD overlays
    displaying RGB camera view, depth map, grasp quality heatmap, and predicted grasps.
    """
    def __init__(self):
        # Color definitions (BGR for OpenCV)
        self.COLOR_TOP_GRASP = (0, 255, 0)     # Bright Green for Top-1
        self.COLOR_ALT_GRASP = (0, 215, 255)   # Gold/Yellow for candidates
        self.COLOR_CENTER = (0, 0, 255)        # Red center point

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
            # Gripper fingers colored cyan
            cv2.line(vis, p1_l, p2_l, (255, 255, 0), thickness + 1)
            cv2.line(vis, p1_r, p2_r, (255, 255, 0), thickness + 1)

            # 3. Draw center point
            c_x, c_y = int(round(grasp.u)), int(round(grasp.v))
            cv2.circle(vis, (c_x, c_y), 4, self.COLOR_CENTER, -1)

            # 4. Text badge on Top-1 grasp
            if is_top:
                badge = f"Q:{grasp.quality:.2f} | {grasp.angle_deg:+.0f}deg"
                cv2.putText(vis, badge, (c_x - 30, c_y - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.putText(vis, badge, (c_x - 30, c_y - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200), 1, cv2.LINE_AA)

        return vis

    def create_dashboard(self, rgb, depth, heatmaps, grasps, telemetry):
        """
        Constructs a 2x2 multi-panel vision-guided grasp dashboard:
        [Panel 1: RGB + Grasp Boxes]   [Panel 2: Colorized Metric Depth]
        [Panel 3: Grasp Quality (Q)]   [Panel 4: Orientation Angle Map]
        """
        h, w = rgb.shape[:2]

        # Panel 1: RGB with Grasp Overlays
        panel1 = self.draw_grasp_overlay(rgb, grasps)

        # Panel 2: Depth map colorized (Turbo / Jet)
        d = depth.copy()
        d_norm = ((d - np.nanmin(d)) / (np.nanmax(d) - np.nanmin(d) + 1e-6) * 255).astype(np.uint8)
        panel2 = cv2.applyColorMap(d_norm, cv2.COLORMAP_TURBO)

        # Panel 3: Grasp Quality Map
        q_norm = (np.clip(heatmaps['q_map'], 0.0, 1.0) * 255).astype(np.uint8)
        panel3 = cv2.applyColorMap(q_norm, cv2.COLORMAP_INFERNO)
        # Mark detected grasp peak centers on Q map
        for g in grasps[:5]:
            cv2.circle(panel3, (int(g.u), int(g.v)), 4, (0, 255, 255), -1)

        # Panel 4: Orientation Angle Map [-pi/2, pi/2] -> [0, 255]
        ang = heatmaps['ang_map']
        ang_norm = ((ang + np.pi/2.0) / np.pi * 255.0).astype(np.uint8)
        panel4 = cv2.applyColorMap(ang_norm, cv2.COLORMAP_HSV)

        # Add panel title labels
        for p, title in zip([panel1, panel2, panel3, panel4],
                            ["1. RGB Perception & Predicted Grasps",
                             "2. Metric Depth Field",
                             "3. CNN Grasp Quality Map Q(u, v)",
                             "4. Gripper Angle Map theta(u, v)"]):
            cv2.rectangle(p, (0, 0), (w, 24), (20, 20, 20), -1)
            cv2.putText(p, title, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # Assemble 2x2 grid
        top_row = np.hstack([panel1, panel2])
        bottom_row = np.hstack([panel3, panel4])
        grid = np.vstack([top_row, bottom_row])

        # Add top HUD banner with live CPS telemetry
        hud_h = 45
        hud = np.full((hud_h, grid.shape[1], 3), 25, dtype=np.uint8)

        lat = telemetry.get('inference_time_ms', 0.0)
        fps = telemetry.get('fps', 0.0)
        dev = telemetry.get('device', 'cpu')
        top_q = telemetry.get('top_score', 0.0)

        t1 = f"CPS DELIVERABLE 1: REAL-TIME VISION-GUIDED GRASP DETECTION"
        t2 = f"Latency: {lat:.1f} ms | Throughput: {fps:.0f} FPS | Device: {dev.upper()} | Best Q: {top_q:.3f}"
        if len(grasps) > 0:
            top_g = grasps[0]
            if top_g.world_coords is not None:
                wx, wy, wz = top_g.world_coords
                t2 += f" | World 3D: [{wx:.2f}, {wy:.2f}, {wz:.2f}]m | Yaw: {top_g.angle_deg:+.1f}deg"

        cv2.putText(hud, t1, (12, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 200), 1, cv2.LINE_AA)
        cv2.putText(hud, t2, (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)

        dashboard = np.vstack([hud, grid])
        return dashboard

    def save_report_figure(self, rgb, depth, heatmaps, grasps, telemetry, save_path="outputs/deliverable1_demo.png"):
        """
        Saves high-resolution publication-quality figure to disk.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        dashboard = self.create_dashboard(rgb, depth, heatmaps, grasps, telemetry)
        # OpenCV uses BGR, write image directly
        cv2.imwrite(save_path, cv2.cvtColor(dashboard, cv2.COLOR_RGB2BGR))
        print(f"Saved Deliverable 1 demonstration figure to: {save_path}")
        return save_path
