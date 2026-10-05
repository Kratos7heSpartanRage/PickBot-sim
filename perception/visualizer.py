import numpy as np
import cv2
import os


class Visualizer:
    """
    State-of-the-Art Perception HUD & Visualization Engine.
    Generates high-definition, publication-grade 4-panel dashboards and
    real-time OpenCV overlays for robotic vision grasping.
    """
    def __init__(self):
        # Professional BGR Color Palette
        self.C_TOP_GRASP = (102, 255, 0)      # Neon Lime (Top-1 Grasp)
        self.C_ALT_GRASP = (0, 200, 255)      # Amber Gold (Alternatives)
        self.C_JAW       = (255, 230, 0)      # Cyan parallel finger contacts
        self.C_CENTER    = (0, 70, 255)       # Coral target reticle
        self.C_WHITE     = (250, 250, 250)
        self.C_DARK_BG   = (16, 20, 28)       # Deep slate-black
        self.C_ACCENT    = (255, 180, 0)      # Cyan highlight
        self.C_GREEN     = (80, 230, 90)

    def draw_grasp_overlay(self, img, grasps, top_k=5):
        """
        Renders high-precision grasp polygons, parallel jaw pads,
        reticle target centers, and sleek score badges.
        """
        vis = img.copy()
        if len(vis.shape) == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

        # Draw candidate grasps first (background)
        for idx in reversed(range(min(top_k, len(grasps)))):
            grasp = grasps[idx]
            is_top = (idx == 0)
            color = self.C_TOP_GRASP if is_top else self.C_ALT_GRASP
            thickness = 2 if is_top else 1

            # 1. Grasp rectangle boundary
            pts = grasp.get_rect_points(jaw_length=16.0).astype(np.int32)
            cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)

            # 2. Parallel jaw contact surfaces (fingers)
            left_jaw, right_jaw, _ = grasp.get_jaw_lines(jaw_length=16.0)
            p1_l, p2_l = tuple(np.int32(left_jaw[0])), tuple(np.int32(left_jaw[1]))
            p1_r, p2_r = tuple(np.int32(right_jaw[0])), tuple(np.int32(right_jaw[1]))
            cv2.line(vis, p1_l, p2_l, self.C_JAW, thickness + 2, lineType=cv2.LINE_AA)
            cv2.line(vis, p1_r, p2_r, self.C_JAW, thickness + 2, lineType=cv2.LINE_AA)

            # 3. Center target reticle
            c_x, c_y = int(round(grasp.u)), int(round(grasp.v))
            if is_top:
                cv2.circle(vis, (c_x, c_y), 6, (0, 0, 0), -1, lineType=cv2.LINE_AA)
                cv2.circle(vis, (c_x, c_y), 5, self.C_TOP_GRASP, -1, lineType=cv2.LINE_AA)
                cv2.circle(vis, (c_x, c_y), 2, (0, 0, 0), -1, lineType=cv2.LINE_AA)
            else:
                cv2.circle(vis, (c_x, c_y), 3, self.C_ALT_GRASP, -1, lineType=cv2.LINE_AA)

            # 4. Floating Score Badge for Top-1
            if is_top:
                badge_text = f"TOP-1  Q:{grasp.quality:.2f} | {grasp.angle_deg:+.0f}deg"
                (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_DUPLEX, 0.40, 1)
                bx = max(4, min(c_x - tw // 2, vis.shape[1] - tw - 12))
                by = max(th + 14, c_y - 18)

                # Badge pill background
                cv2.rectangle(vis, (bx - 4, by - th - 5), (bx + tw + 4, by + 4), (15, 20, 26), -1)
                cv2.rectangle(vis, (bx - 4, by - th - 5), (bx + tw + 4, by + 4), self.C_TOP_GRASP, 1)
                cv2.putText(vis, badge_text, (bx, by), cv2.FONT_HERSHEY_DUPLEX, 0.40, self.C_WHITE, 1, cv2.LINE_AA)

        return vis

    def create_dashboard(self, rgb, depth, heatmaps, grasps, telemetry):
        """
        Constructs a cyber-industrial 2x2 perception dashboard:
          [Panel 1: RGB Perception & Grasps]  [Panel 2: Metric Depth Field]
          [Panel 3: CNN Quality Map Q(u, v)]   [Panel 4: Gripper Angle theta]
        Surrounded by a real-time HUD header and interactive bottom status bar.
        """
        h, w = rgb.shape[:2]

        # --- Panel 1: RGB + Grasp Overlays ---
        panel1 = self.draw_grasp_overlay(rgb, grasps)

        # --- Panel 2: Metric Depth Field (Turbo Colormap) ---
        d = depth.copy()
        d_min, d_max = np.nanmin(d), np.nanmax(d)
        d_norm = ((d - d_min) / (d_max - d_min + 1e-6) * 255.0).astype(np.uint8)
        panel2 = cv2.applyColorMap(d_norm, cv2.COLORMAP_TURBO)
        # Depth range badge
        range_str = f"Z: {d_min:.2f}m - {d_max:.2f}m"
        cv2.rectangle(panel2, (w - 130, h - 22), (w - 4, h - 4), (16, 20, 26), -1)
        cv2.putText(panel2, range_str, (w - 124, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 220, 220), 1, cv2.LINE_AA)

        # --- Panel 3: Grasp Quality Map Q(u, v) (Inferno) ---
        q_norm = (np.clip(heatmaps['q_map'], 0.0, 1.0) * 255.0).astype(np.uint8)
        panel3 = cv2.applyColorMap(q_norm, cv2.COLORMAP_INFERNO)
        # Mark local maxima peaks
        for r_idx, g in enumerate(grasps[:5]):
            gx, gy = int(round(g.u)), int(round(g.v))
            cv2.circle(panel3, (gx, gy), 6, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(panel3, f"#{r_idx+1}", (gx + 8, gy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        # --- Panel 4: Gripper Orientation Map Theta (HSV) ---
        ang = heatmaps['ang_map']
        ang_norm = ((ang + np.pi / 2.0) / np.pi * 255.0).astype(np.uint8)
        panel4 = cv2.applyColorMap(ang_norm, cv2.COLORMAP_HSV)

        # --- Style Each Panel with Modern Dark Header ---
        headers = [
            ("1. RGB CAMERA // PREDICTED GRASPS", (102, 255, 0)),
            ("2. METRIC DEPTH SENSOR FIELD", (255, 180, 0)),
            ("3. CNN GRASP QUALITY MAP Q(u,v)", (0, 140, 255)),
            ("4. GRIPPER ANGLE MAP THETA(u,v)", (220, 80, 255)),
        ]
        panels = [panel1, panel2, panel3, panel4]
        for p, (title, tag_color) in zip(panels, headers):
            # Top card banner
            cv2.rectangle(p, (0, 0), (w, 24), (16, 20, 26), -1)
            # Accent tag
            cv2.rectangle(p, (6, 6), (12, 18), tag_color, -1)
            cv2.putText(p, title, (18, 17), cv2.FONT_HERSHEY_DUPLEX, 0.38, (240, 240, 240), 1, cv2.LINE_AA)
            # Thin divider
            cv2.line(p, (0, 24), (w, 24), (35, 45, 60), 1)

        # Assemble 2x2 Grid with 2px borders
        top_row = np.hstack([panel1, panel2])
        bottom_row = np.hstack([panel3, panel4])
        grid = np.vstack([top_row, bottom_row])
        grid_w = grid.shape[1]

        # --- TOP HUD HEADER BAR (60px) ---
        hud_h = 60
        hud = np.full((hud_h, grid_w, 3), 16, dtype=np.uint8)
        # Background gradient line
        cv2.line(hud, (0, hud_h - 1), (grid_w, hud_h - 1), (45, 65, 90), 1)

        # Title & Subtitle
        cv2.putText(hud, "PICKBOT ROBOTIC CELL // REAL-TIME GRASP SYNTHESIS",
                    (14, 24), cv2.FONT_HERSHEY_DUPLEX, 0.52, (0, 230, 255), 1, cv2.LINE_AA)

        lat = telemetry.get('inference_time_ms', 0.0)
        fps = telemetry.get('fps', 0.0)
        dev = str(telemetry.get('device', 'cpu')).upper()
        top_q = telemetry.get('top_score', 0.0)
        num_cand = len(grasps)

        # Pills/Badges in Header
        pills = [
            (f"STATUS: ONLINE", (0, 200, 80)),
            (f"DEVICE: {dev}", (220, 160, 40)),
            (f"LATENCY: {lat:.1f}ms ({fps:.0f} FPS)", (180, 100, 255)),
            (f"BEST Q: {top_q:.2f}", (0, 215, 255)),
            (f"TARGETS: {num_cand}", (240, 240, 240)),
        ]
        px = 14
        for text, pcolor in pills:
            (pw, ph), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
            cv2.rectangle(hud, (px, 34), (px + pw + 12, 52), (28, 35, 48), -1)
            cv2.rectangle(hud, (px, 34), (px + pw + 12, 52), pcolor, 1)
            cv2.putText(hud, text, (px + 6, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.35, pcolor, 1, cv2.LINE_AA)
            px += pw + 20

        # --- BOTTOM ACTION STATUS FOOTER (26px) ---
        foot_h = 26
        foot = np.full((foot_h, grid_w, 3), 16, dtype=np.uint8)
        cv2.line(foot, (0, 0), (grid_w, 0), (45, 65, 90), 1)
        foot_text = "HOTKEYS: [D] Detect Grasps  |  [Space] Execute Pick-and-Place  |  [R] Respawn Scene  |  [Q] Exit"
        cv2.putText(foot, foot_text, (14, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 185, 200), 1, cv2.LINE_AA)

        dashboard = np.vstack([hud, grid, foot])
        return dashboard

    def save_report_figure(self, rgb, depth, heatmaps, grasps, telemetry, save_path="outputs/deliverable1_demo.png"):
        """Saves high-resolution publication-quality figure to disk."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        dashboard = self.create_dashboard(rgb, depth, heatmaps, grasps, telemetry)
        cv2.imwrite(save_path, cv2.cvtColor(dashboard, cv2.COLOR_RGB2BGR))
        print(f"Saved demonstration figure to: {save_path}")
        return save_path
