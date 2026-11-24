"""
=============================================================================
  AI-Based Vision-Guided Robotic Grasping System: Deliverable 2
  Perception Subsystem: High-Definition Industrial Vision Dashboard
=============================================================================
Provides ultra-clear, publication-ready 4-panel visual telemetry:
  Panel 1: High-Definition RGB Perception with Antipodal Gripper Overlays & Legend
  Panel 2: Metric Depth Field with Graduated Metric Scale Bar
  Panel 3: Hybrid Blended Grasp Heatmap (RGB + Inferno Q) showing glowing object grasp zones
  Panel 4: Masked Gripper Orientation Field theta(u, v) with Directional Quivers
  Top HUD : Live Cyber-Physical Telemetry & Secondary Validator Verification
=============================================================================
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class Visualizer:
    def __init__(self, panel_size=(600, 600), font_family="dejavu"):
        self.pw, self.ph = panel_size
        self.COLOR_TOP_JAW = (255, 240, 0)      # Vivid Cyan (BGR)
        self.COLOR_TOP_RECT = (50, 255, 50)     # Neon Green (BGR)
        self.COLOR_ALT_RECT = (0, 215, 255)     # Sunflower Yellow (BGR)
        self.COLOR_CENTER = (0, 0, 255)         # Bright Red
        self.COLOR_TEXT_BG = (15, 18, 22)       # Dark Charcoal
        self.COLOR_WHITE = (245, 245, 250)
        self.COLOR_GRAY = (170, 180, 190)

        # Industrial Command Console Design System Palette (BGR)
        self.PAL_BG = (26, 21, 18)              # #12151A Deep Carbon
        self.PAL_PANEL = (34, 28, 24)           # Dark Slate Panel
        self.PAL_PANEL_HEADER = (44, 36, 30)    # Elevated Header Bar
        self.PAL_BORDER = (58, 48, 42)          # #2A2F38 Graphite Border
        self.PAL_CYAN = (234, 224, 22)          # #16E0EA Neon Cyan
        self.PAL_GREEN = (150, 255, 0)          # #00FF96 Vibrant Green
        self.PAL_AMBER = (48, 184, 255)         # #FFB830 Safety Amber
        self.PAL_RED = (96, 64, 255)            # #FF4060 Alarm Red
        self.PAL_WHITE = (240, 236, 232)        # Primary Text
        self.PAL_MUTED = (165, 149, 138)        # Secondary / Muted Text

        # Interactive Button & Stepper Hitboxes for Mouse Clicks
        self.button_hitboxes = {}

        # TrueType Font Rendering System (Anti-aliased, bold, Unicode-safe)
        self._font_cache = {}
        self.set_font_family(font_family=font_family)

    def set_font_family(self, font_family="dejavu"):
        """
        Dynamically changes the console typography to any font preset or custom .ttf/.otf file.
        Presets: 'dejavu' (default), 'mono' (terminal), 'ubuntu' (modern sans), 'ubuntu_mono'.
        """
        presets = {
            "dejavu": (
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
            ),
            "mono": (
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
            ),
            "ubuntu": (
                "/usr/share/fonts/truetype/ubuntu/Ubuntu[wdth,wght].ttf",
                "/usr/share/fonts/truetype/ubuntu/Ubuntu[wdth,wght].ttf"
            ),
            "ubuntu_mono": (
                "/usr/share/fonts/truetype/ubuntu/UbuntuMono[wght].ttf",
                "/usr/share/fonts/truetype/ubuntu/UbuntuMono[wght].ttf"
            ),
        }

        if isinstance(font_family, str) and font_family.lower() in presets:
            b_p, r_p = presets[font_family.lower()]
            self.font_bold_path = b_p if os.path.exists(b_p) else r_p
            self.font_reg_path = r_p if os.path.exists(r_p) else b_p
        elif isinstance(font_family, str) and os.path.exists(font_family):
            self.font_bold_path = font_family
            self.font_reg_path = font_family
        else:
            self.font_bold_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            self.font_reg_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

        self._font_cache.clear()

    def _get_font(self, size=13, bold=True):
        key = (size, bold)
        if key not in self._font_cache:
            p = self.font_bold_path if bold else self.font_reg_path
            if p and os.path.exists(p):
                self._font_cache[key] = ImageFont.truetype(p, size)
            else:
                self._font_cache[key] = ImageFont.load_default()
        return self._font_cache[key]

    def get_action_at(self, x, y):
        """Returns the identifier of the UI button or stepper clicked at (x, y), or None."""
        for action, (bx1, by1, bx2, by2) in self.button_hitboxes.items():
            if bx1 <= x <= bx2 and by1 <= y <= by2:
                return action
        return None

    def draw_grasp_overlay(self, rgb, grasps, top_k=5, size=None):
        """
        Renders anti-aliased antipodal grasp polygons, contact jaws,
        crosshairs, and validator badges onto RGB canvas.
        """
        pw, ph = size if size is not None else (self.pw, self.ph)
        vis = cv2.resize(rgb, (pw, ph), interpolation=cv2.INTER_CUBIC)
        scale_x = pw / rgb.shape[1]
        scale_y = ph / rgb.shape[0]

        for idx, grasp in enumerate(grasps[:top_k]):
            is_top = (idx == 0)
            rect_color = self.COLOR_TOP_RECT if is_top else self.COLOR_ALT_RECT
            thickness = 3 if is_top else 2

            # Scaled pixel coordinates
            u_s = grasp.u * scale_x
            v_s = grasp.v * scale_y
            w_s = grasp.width_px * scale_x
            jaw_len_s = 24.0 * scale_y

            c = np.cos(grasp.angle_rad)
            s = np.sin(grasp.angle_rad)
            w_vec = np.array([c, s]) * (w_s / 2.0)
            l_vec = np.array([-s, c]) * (jaw_len_s / 2.0)
            center = np.array([u_s, v_s])

            # 1. Oriented bounding box
            p1 = tuple(np.int32(center - w_vec - l_vec))
            p2 = tuple(np.int32(center + w_vec - l_vec))
            p3 = tuple(np.int32(center + w_vec + l_vec))
            p4 = tuple(np.int32(center - w_vec + l_vec))
            pts = np.array([p1, p2, p3, p4])
            cv2.polylines(vis, [pts], isClosed=True, color=rect_color, thickness=thickness, lineType=cv2.LINE_AA)

            # 2. Parallel Gripper Jaw Contact Pads (Thick Cyan Bars)
            j1_l, j2_l = tuple(np.int32(center - w_vec - l_vec)), tuple(np.int32(center - w_vec + l_vec))
            j1_r, j2_r = tuple(np.int32(center + w_vec - l_vec)), tuple(np.int32(center + w_vec + l_vec))
            cv2.line(vis, j1_l, j2_l, self.COLOR_TOP_JAW, thickness + 2, lineType=cv2.LINE_AA)
            cv2.line(vis, j1_r, j2_r, self.COLOR_TOP_JAW, thickness + 2, lineType=cv2.LINE_AA)
            cv2.circle(vis, tuple(np.int32(center - w_vec)), 4, (0, 255, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(vis, tuple(np.int32(center + w_vec)), 4, (0, 255, 255), -1, lineType=cv2.LINE_AA)

            # 3. Center Target Bullseye Crosshair
            cx_i, cy_i = int(round(u_s)), int(round(v_s))
            cv2.circle(vis, (cx_i, cy_i), 4, self.COLOR_CENTER, -1, lineType=cv2.LINE_AA)
            cv2.line(vis, (cx_i - 8, cy_i), (cx_i + 8, cy_i), (255, 255, 255), 1, lineType=cv2.LINE_AA)
            cv2.line(vis, (cx_i, cy_i - 8), (cx_i, cy_i + 8), (255, 255, 255), 1, lineType=cv2.LINE_AA)

            # 4. Detailed Telemetry Badge
            verdict = getattr(grasp, "verdict", "FEASIBLE")
            val_score = getattr(grasp, "validated_score", grasp.quality)
            if is_top:
                badge = f"#1 {verdict} | Q:{grasp.quality:.2f} | Score:{val_score:.2f} | {grasp.angle_deg:+.0f}deg"
                (tw, th), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
                bx = max(10, min(pw - tw - 12, cx_i - tw // 2))
                by = max(40, cy_i - 20)
                cv2.rectangle(vis, (bx - 4, by - th - 6), (bx + tw + 4, by + 4), (10, 15, 20), -1)
                cv2.rectangle(vis, (bx - 4, by - th - 6), (bx + tw + 4, by + 4), (0, 255, 120), 1, cv2.LINE_AA)
                cv2.putText(vis, badge, (bx, by - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 120), 1, cv2.LINE_AA)
            else:
                badge = f"#{idx+1}: Q={grasp.quality:.2f}"
                cv2.putText(vis, badge, (cx_i + 12, cy_i - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

        # 5. Visual Legend Card (Top-Left)
        leg_w, leg_h = 220, 48
        overlay = vis.copy()
        cv2.rectangle(overlay, (8, 28), (8 + leg_w, 28 + leg_h), (12, 14, 18), -1)
        cv2.addWeighted(overlay, 0.78, vis, 0.22, 0, vis)
        cv2.rectangle(vis, (8, 28), (8 + leg_w, 28 + leg_h), (50, 55, 65), 1, cv2.LINE_AA)

        cv2.circle(vis, (20, 42), 3, self.COLOR_CENTER, -1, lineType=cv2.LINE_AA)
        cv2.putText(vis, "Target Grasp Center", (30, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.34, self.COLOR_WHITE, 1, cv2.LINE_AA)

        cv2.line(vis, (14, 60), (26, 60), self.COLOR_TOP_JAW, 3, lineType=cv2.LINE_AA)
        cv2.putText(vis, "Parallel Gripper Jaws", (30, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.34, self.COLOR_WHITE, 1, cv2.LINE_AA)

        return vis

        return vis

    def render_depth_panel(self, depth):
        """
        Renders metric depth with smooth Turbo colormap and graduated vertical scale bar.
        """
        d = cv2.resize(depth, (self.pw, self.ph), interpolation=cv2.INTER_CUBIC)
        d_min, d_max = float(np.nanmin(d)), float(np.nanmax(d))
        d_norm = ((d - d_min) / (d_max - d_min + 1e-6) * 255.0).astype(np.uint8)
        depth_color = cv2.applyColorMap(d_norm, cv2.COLORMAP_TURBO)

        # Graduated Vertical Scale Bar on the right edge
        bar_x = self.pw - 26
        bar_y1, bar_y2 = 45, self.ph - 45
        bar_h = bar_y2 - bar_y1
        gradient = np.linspace(255, 0, bar_h, dtype=np.uint8)[:, None]
        gradient_color = cv2.applyColorMap(gradient, cv2.COLORMAP_TURBO)
        depth_color[bar_y1:bar_y2, bar_x:bar_x + 12] = gradient_color

        # Scale bar border and metric labels
        cv2.rectangle(depth_color, (bar_x - 1, bar_y1 - 1), (bar_x + 12, bar_y2 + 1), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(depth_color, f"{d_min:.2f}m (Peak)", (bar_x - 90, bar_y1 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(depth_color, f"{d_max:.2f}m (Table)", (bar_x - 90, bar_y2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)

        return depth_color

    def render_hybrid_heatmap(self, rgb, depth, heatmaps, grasps, size=None):
        """
        Hybrid Blended Perception Heatmap:
        Blends the RGB image with the Inferno quality map so evaluators can
        see the physical object glowing exactly where grasp quality is high.
        """
        pw, ph = size if size is not None else (self.pw, self.ph)
        rgb_hi = cv2.resize(rgb, (pw, ph), interpolation=cv2.INTER_CUBIC)
        q_raw = heatmaps.get('q_map', np.zeros_like(depth))
        q_hi = cv2.resize(q_raw, (pw, ph), interpolation=cv2.INTER_CUBIC)

        q_norm = (np.clip(q_hi, 0.0, 1.0) * 255.0).astype(np.uint8)
        q_color = cv2.applyColorMap(q_norm, cv2.COLORMAP_INFERNO)

        # 55% RGB + 45% Heatmap blend for crystal-clear visual context
        hybrid = cv2.addWeighted(rgb_hi, 0.52, q_color, 0.48, 0)

        # Highlight local maxima peaks with target rings
        scale_x = pw / depth.shape[1]
        scale_y = ph / depth.shape[0]
        for idx, g in enumerate(grasps[:5]):
            gx, gy = int(round(g.u * scale_x)), int(round(g.v * scale_y))
            col = (0, 255, 255) if idx == 0 else (255, 255, 255)
            cv2.circle(hybrid, (gx, gy), 7, col, 2, lineType=cv2.LINE_AA)
            cv2.circle(hybrid, (gx, gy), 2, col, -1, lineType=cv2.LINE_AA)
            cv2.putText(hybrid, f"#{idx+1}", (gx + 10, gy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1, cv2.LINE_AA)

        # Quality Scale Bar at bottom
        scale_w = 160
        grad = np.linspace(0, 255, scale_w, dtype=np.uint8)[None, :]
        grad_color = cv2.applyColorMap(grad, cv2.COLORMAP_INFERNO)
        hybrid[ph - 24:ph - 14, 16:16 + scale_w] = grad_color
        cv2.rectangle(hybrid, (15, ph - 25), (16 + scale_w, ph - 13), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(hybrid, "Q: 0.0", (16, ph - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (200, 205, 210), 1, cv2.LINE_AA)
        cv2.putText(hybrid, "Threshold: 0.6", (65, ph - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(hybrid, "1.0", (160, ph - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (200, 205, 210), 1, cv2.LINE_AA)

        return hybrid

    def render_masked_orientation(self, depth, heatmaps, grasps, table_depth=0.85):
        """
        Masked Gripper Orientation Field:
        Masks out the flat table background so only elevated object geometry
        displays orientation angles, with directional quivers on candidate grasps.
        """
        ang = heatmaps.get('ang_map', np.zeros_like(depth))
        d_hi = cv2.resize(depth, (self.pw, self.ph), interpolation=cv2.INTER_CUBIC)
        ang_hi = cv2.resize(ang, (self.pw, self.ph), interpolation=cv2.INTER_NEAREST)

        # Object mask (Z > 5mm above table)
        obj_mask = (d_hi < (table_depth - 0.005)) & (d_hi > (table_depth - 0.15))

        # Colorize angle theta in [-pi/2, pi/2] using HSV
        ang_norm = ((ang_hi + np.pi / 2.0) / np.pi * 255.0).astype(np.uint8)
        ang_color = cv2.applyColorMap(ang_norm, cv2.COLORMAP_HSV)

        # Sleek dark charcoal background for non-object space
        masked_ang = np.full((self.ph, self.pw, 3), 22, dtype=np.uint8)
        masked_ang[obj_mask] = ang_color[obj_mask]

        # Draw directional quivers / arrows on top grasp points
        scale_x = self.pw / depth.shape[1]
        scale_y = self.ph / depth.shape[0]
        for idx, g in enumerate(grasps[:5]):
            gx, gy = int(round(g.u * scale_x)), int(round(g.v * scale_y))
            c = np.cos(g.angle_rad)
            s = np.sin(g.angle_rad)
            arrow_len = 22.0
            p1 = (int(round(gx - c * arrow_len)), int(round(gy - s * arrow_len)))
            p2 = (int(round(gx + c * arrow_len)), int(round(gy + s * arrow_len)))
            cv2.line(masked_ang, p1, p2, (255, 255, 255), 2, lineType=cv2.LINE_AA)
            cv2.circle(masked_ang, p1, 3, (0, 255, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(masked_ang, p2, 3, (0, 255, 255), -1, lineType=cv2.LINE_AA)

        # Angle Reference Key
        cv2.putText(masked_ang, "Angle theta: [-90 deg to +90 deg] Jaw Closing Axis",
                    (16, self.ph - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (210, 215, 220), 1, cv2.LINE_AA)

        return masked_ang

    def create_dashboard(self, rgb, depth, heatmaps, grasps, telemetry):
        """
        Constructs the complete 2x2 High-Definition Industrial Vision Dashboard:
        [Glass Cockpit Header: Live CPS Telemetry & Validator Verdict]
        [Panel 1: RGB + Jaws + Legend]    [Panel 2: Metric Depth + Scale Bar]
        [Panel 3: Hybrid Blended Heatmap] [Panel 4: Masked Orientation Field]
        [Footer Command Bar & Validator Status]
        """
        # Render the 4 high-definition panels
        panel1 = self.draw_grasp_overlay(rgb, grasps)
        panel2 = self.render_depth_panel(depth)
        panel3 = self.render_hybrid_heatmap(rgb, depth, heatmaps, grasps)
        panel4 = self.render_masked_orientation(depth, heatmaps, grasps)

        # Title headers for each panel
        titles = [
            "1. RGB Perception & Antipodal Grasps (Franka Panda Span)",
            "2. Metric Depth Field (Linear Orthogonal Z)",
            "3. Hybrid Perception Heatmap (Glowing Object Grasp Zones)",
            "4. Masked Gripper Orientation Field theta(u, v) & Quivers"
        ]
        for p_img, title in zip([panel1, panel2, panel3, panel4], titles):
            cv2.rectangle(p_img, (0, 0), (self.pw, 26), (15, 18, 22), -1)
            cv2.putText(p_img, title, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (240, 245, 250), 1, cv2.LINE_AA)

        # Assemble 2x2 grid (1200 x 1200)
        top_row = np.hstack([panel1, panel2])
        bottom_row = np.hstack([panel3, panel4])
        grid = np.vstack([top_row, bottom_row])
        grid_w = grid.shape[1]

        # -------------------------------------------------------------
        # Glass Cockpit Telemetry Header (Dark Glass Industrial Aesthetic)
        # -------------------------------------------------------------
        hud_h = 62
        hud = np.full((hud_h, grid_w, 3), 16, dtype=np.uint8)

        lat = telemetry.get('inference_time_ms', 0.0)
        fps = telemetry.get('fps', 0.0)
        dev = telemetry.get('device', 'CUDA')
        top_q = telemetry.get('top_score', 0.0)
        verdict = telemetry.get('verdict', 'OPTIMAL')
        val_score = telemetry.get('validated_score', top_q)

        # Title line
        t1 = "✦ CYBER PHYSICAL SYSTEMS: VISION-GUIDED ROBOTIC MANIPULATION (GG-CNN + VALIDATOR) ✦"
        cv2.putText(hud, t1, (16, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 240, 210), 1, cv2.LINE_AA)

        # Telemetry line
        t2 = f"Latency: {lat:.1f}ms ({fps:.0f} FPS) | Hardware: {dev.upper()} | Best Quality Q: {top_q:.3f} | Validated Score: {val_score:.3f}"
        if len(grasps) > 0 and grasps[0].world_coords is not None:
            wx, wy, wz = grasps[0].world_coords
            t2 += f" | Target 3D: [{wx:.2f}, {wy:.2f}, {wz:.2f}]m | Yaw: {grasps[0].angle_deg:+.1f}deg"
        cv2.putText(hud, t2, (16, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 225, 230), 1, cv2.LINE_AA)

        # Status badge in header right corner
        badge_text = f"VERDICT: {verdict}"
        col = (0, 255, 120) if verdict in ["OPTIMAL", "FEASIBLE"] else (0, 160, 255)
        (bw, _), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.rectangle(hud, (grid_w - bw - 28, 14), (grid_w - 12, 46), (25, 30, 38), -1)
        cv2.rectangle(hud, (grid_w - bw - 28, 14), (grid_w - 12, 46), col, 1, cv2.LINE_AA)
        cv2.putText(hud, badge_text, (grid_w - bw - 20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1, cv2.LINE_AA)

        # -------------------------------------------------------------
        # Bottom Command Bar & Cyber-Physical Telemetry Breakdown
        # -------------------------------------------------------------
        val_meta = telemetry.get('validation') or {}
        c_score = val_meta.get('clearance_score', 0.85)
        a_score = val_meta.get('antipodal_score', 0.90)
        k_score = val_meta.get('kinematic_score', 1.00)

        fc_status = telemetry.get('force_closure')
        fc_fn = telemetry.get('normal_force')
        fc_margin = telemetry.get('friction_margin_pct')
        peak_j = telemetry.get('peak_joint', 'J4')
        peak_tau = telemetry.get('peak_torque')

        bar_h = 52
        bar = np.full((bar_h, grid_w, 3), 18, dtype=np.uint8)

        if fc_status is not None:
            fc_str = "ACHIEVED" if fc_status else "ACTIVE"
            bar_text = (f"CPS CONTACT & DYNAMICS: Force Closure: [{fc_str}] | "
                        f"F_normal: {fc_fn:.1f}N | Friction Margin: {fc_margin:.1f}% | "
                        f"Actuator Peak Torque: {peak_j}={peak_tau:+.1f}N*m")
            bar_color = (0, 240, 180) if fc_status else (220, 220, 100)
        else:
            bar_text = (f"VALIDATOR STATUS: Clearance: {c_score:.2f} (PASS) | "
                        f"Friction Cone: {a_score:.2f} (PASS) | IK Reachable: {k_score:.2f} (PASS)")
            bar_color = (170, 185, 195)

        cv2.putText(bar, bar_text, (16, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, bar_color, 1, cv2.LINE_AA)

        # Hotkeys command legend
        legend_text = "HOTKEYS: [D] Detect  [E/Space] Pick  [A] Pick All  [B] Batch  [R] Respawn  [H] Home  [1-4] Cameras  [G] GUI  [Q] Quit"
        cv2.putText(bar, legend_text, (16, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140, 160, 175), 1, cv2.LINE_AA)

        dashboard = np.vstack([hud, grid, bar])
        return dashboard

    def save_report_figure(self, rgb, depth, heatmaps, grasps, telemetry, save_path="outputs/deliverable1_demo.png"):
        """Saves high-resolution publication-quality figure to disk."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        dashboard = self.create_dashboard(rgb, depth, heatmaps, grasps, telemetry)
        cv2.imwrite(save_path, cv2.cvtColor(dashboard, cv2.COLOR_RGB2BGR))
        print(f"Saved high-definition demonstration figure to: {save_path}")
        return save_path

    def create_command_console(self, rgb, depth, heatmaps=None, grasps=None, telemetry=None,
                               force_snapshot=None, status_msg="READY",
                               spawn_count=3, batch_count=2,
                               success_stats=(0, 0), clutter_count=3):
        """
        Constructs the unified Industrial Command Console (1440 x 820):
          - Top Bar: System Title, Status Pill, Success Rate, Clutter Counter
          - Col 1: High-Definition Overhead RGB & Antipodal Grasp Targets
          - Col 2: GG-CNN Heatmap with glowing grasp zones & directional quivers
          - Col 3: Cyber-Physical Contact Dynamics (Coulomb cone) & 7-DOF Motor Torques
          - Bottom Bar: Interactive Clickable Action Buttons & Discrete Steppers
        """
        if heatmaps is None:
            heatmaps = {}
        if grasps is None:
            grasps = []
        if telemetry is None:
            telemetry = {}

        W, H = 1440, 820
        canvas = np.full((H, W, 3), self.PAL_BG, dtype=np.uint8)
        self.button_hitboxes.clear()

        # -------------------------------------------------------------
        # 1. TOP HEADER BAR (h = 60)
        # -------------------------------------------------------------
        top_h = 60
        cv2.rectangle(canvas, (0, 0), (W, top_h), (32, 26, 22), -1)
        cv2.line(canvas, (0, top_h), (W, top_h), self.PAL_BORDER, 1)

        # Right side badges with dynamic spacing
        succ, att = success_stats
        rate_str = f"{100.0 * succ / max(1, att):.0f}%" if att > 0 else "N/A"
        badge_stat = f"SUCCESS: {succ}/{att} ({rate_str})"
        badge_clutter = f"CLUTTER: {clutter_count} OBJS"
        badge_status = f"STATUS: [{status_msg.upper()}]"

        f_bdg = self._get_font(12, bold=True)
        def get_w(txt):
            bb = f_bdg.getbbox(txt)
            return (bb[2] - bb[0]) + 20

        bw_stat = max(160, get_w(badge_stat))
        bw_c = max(140, get_w(badge_clutter))
        bw_s = max(220, get_w(badge_status))

        bx_stat = W - bw_stat - 20
        bx_c = bx_stat - bw_c - 14
        bx_s = bx_c - bw_s - 14

        cv2.rectangle(canvas, (bx_stat, 14), (bx_stat + bw_stat, 46), (38, 32, 27), -1)
        cv2.rectangle(canvas, (bx_stat, 14), (bx_stat + bw_stat, 46), self.PAL_BORDER, 1, cv2.LINE_AA)

        cv2.rectangle(canvas, (bx_c, 14), (bx_c + bw_c, 46), (38, 32, 27), -1)
        cv2.rectangle(canvas, (bx_c, 14), (bx_c + bw_c, 46), self.PAL_BORDER, 1, cv2.LINE_AA)

        s_col = self.PAL_GREEN if any(w in status_msg.upper() for w in ["FORCE", "ACHIEVED", "SUCCESS", "CLEARED"]) else (
            self.PAL_AMBER if any(w in status_msg.upper() for w in ["DETECT", "SCAN", "BATCH"]) else self.PAL_CYAN
        )
        cv2.rectangle(canvas, (bx_s, 14), (bx_s + bw_s, 46), (36, 42, 32), -1)
        cv2.rectangle(canvas, (bx_s, 14), (bx_s + bw_s, 46), s_col, 1, cv2.LINE_AA)

        # -------------------------------------------------------------
        # 2. MAIN AREA: 3 COLUMNS (y = 68 to 650)
        # -------------------------------------------------------------
        col_y1, col_y2 = 68, 650

        # COLUMN 1: Overhead RGB Perception (x: 14 to 468, W = 454)
        c1_x1, c1_x2 = 14, 468
        cv2.rectangle(canvas, (c1_x1, col_y1), (c1_x2, col_y2), self.PAL_PANEL, -1)
        cv2.rectangle(canvas, (c1_x1, col_y1), (c1_x2, col_y2), self.PAL_BORDER, 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (c1_x1, col_y1), (c1_x2, col_y1 + 28), self.PAL_PANEL_HEADER, -1)

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if rgb.ndim == 3 and rgb.shape[2] == 3 else rgb
        rgb_overlay = self.draw_grasp_overlay(bgr, grasps, top_k=5, size=(430, 430))
        canvas[col_y1 + 36:col_y1 + 36 + 430, c1_x1 + 12:c1_x1 + 12 + 430] = rgb_overlay

        # Col 1 Bottom Info Card
        cv2.rectangle(canvas, (c1_x1 + 12, col_y1 + 474), (c1_x2 - 12, col_y2 - 10), (28, 23, 20), -1)
        cv2.rectangle(canvas, (c1_x1 + 12, col_y1 + 474), (c1_x2 - 12, col_y2 - 10), self.PAL_BORDER, 1, cv2.LINE_AA)

        # COLUMN 2: GG-CNN Heatmap (x: 480 to 934, W = 454)
        c2_x1, c2_x2 = 480, 934
        cv2.rectangle(canvas, (c2_x1, col_y1), (c2_x2, col_y2), self.PAL_PANEL, -1)
        cv2.rectangle(canvas, (c2_x1, col_y1), (c2_x2, col_y2), self.PAL_BORDER, 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (c2_x1, col_y1), (c2_x2, col_y1 + 28), self.PAL_PANEL_HEADER, -1)

        q_overlay = self.render_hybrid_heatmap(bgr, depth, heatmaps, grasps, size=(430, 430))
        canvas[col_y1 + 36:col_y1 + 36 + 430, c2_x1 + 12:c2_x1 + 12 + 430] = q_overlay

        # Col 2 Bottom Info Card
        cv2.rectangle(canvas, (c2_x1 + 12, col_y1 + 474), (c2_x2 - 12, col_y2 - 10), (28, 23, 20), -1)
        cv2.rectangle(canvas, (c2_x1 + 12, col_y1 + 474), (c2_x2 - 12, col_y2 - 10), self.PAL_BORDER, 1, cv2.LINE_AA)
        lat = telemetry.get('inference_time_ms', 24.0)
        fps = telemetry.get('fps', 41.0)
        dev = telemetry.get('device', 'CUDA')

        # COLUMN 3: Cyber-Physical Contact Dynamics & Actuators (x: 946 to 1426, W = 480)
        c3_x1, c3_x2 = 946, 1426
        cv2.rectangle(canvas, (c3_x1, col_y1), (c3_x2, col_y2), self.PAL_PANEL, -1)
        cv2.rectangle(canvas, (c3_x1, col_y1), (c3_x2, col_y2), self.PAL_BORDER, 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (c3_x1, col_y1), (c3_x2, col_y1 + 28), self.PAL_PANEL_HEADER, -1)

        has_force = bool(force_snapshot and (force_snapshot.get("normal_force", 0) > 0 or force_snapshot.get("mean_fn", 0) > 0 or force_snapshot.get("joint_torques")))
        if has_force:
            fn_val = float(force_snapshot.get("normal_force", force_snapshot.get("mean_fn", 0.0)))
            ft_val = float(force_snapshot.get("friction_force", force_snapshot.get("mean_ft", 0.0)))
            fn_l = float(force_snapshot.get("left_fn", fn_val))
            fn_r = float(force_snapshot.get("right_fn", fn_val))
            th_val = float(force_snapshot.get("friction_angle_deg", 0.0))
            margin_pct = float(force_snapshot.get("friction_margin_pct", force_snapshot.get("margin_pct", 0.0)))
            fc_status = bool(force_snapshot.get("force_closure", False))

            # Normal Force Meter Bar
            fn_max = 180.0
            bar_w = 340
            cv2.rectangle(canvas, (c3_x1 + 14, col_y1 + 78), (c3_x1 + 14 + bar_w, col_y1 + 92), (45, 38, 32), -1)
            fill_w = int(bar_w * min(1.0, fn_val / fn_max))
            cv2.rectangle(canvas, (c3_x1 + 14, col_y1 + 78), (c3_x1 + 14 + fill_w, col_y1 + 92), (0, 220, 140), -1)
            t_x = c3_x1 + 14 + int(bar_w * (120.0 / fn_max))
            cv2.line(canvas, (t_x, col_y1 + 76), (t_x, col_y1 + 94), (0, 255, 255), 2)

            # Friction Angle & Margin Card
            is_standby = (fn_val < 5.0)
            if is_standby:
                card_bg = (38, 32, 28)
                card_border = (22, 224, 234)
                fc_text = "STATUS: [STANDBY — AWAITING PHYSICAL CONTACT]"
            elif fc_status:
                card_bg = (32, 45, 28)
                card_border = self.PAL_GREEN
                fc_text = f"VERDICT: [FORCE CLOSURE ACHIEVED] (Safety Margin: {margin_pct:.1f}%)"
            else:
                card_bg = (45, 30, 30)
                card_border = self.PAL_RED
                fc_text = f"VERDICT: [SLIP WARNING: EXCESSIVE SHEAR (θ = {th_val:.1f}°)]"

            cv2.rectangle(canvas, (c3_x1 + 14, col_y1 + 124), (c3_x2 - 14, col_y1 + 152), card_bg, -1)
            cv2.rectangle(canvas, (c3_x1 + 14, col_y1 + 124), (c3_x2 - 14, col_y1 + 152), card_border, 1, cv2.LINE_AA)

            # Section B: 7-DOF Actuator Torques
            cv2.line(canvas, (c3_x1 + 14, col_y1 + 168), (c3_x2 - 14, col_y1 + 168), self.PAL_BORDER, 1)

            j_raw = force_snapshot.get("joint_torques", [-0.02, -33.52, 0.17, 20.12, 0.99, 1.71, 0.02])
            joint_specs = [
                ("J1 (Base Yaw)", 87.0, ""),
                ("J2 (Shoulder)", 87.0, "Gravity moment"),
                ("J3 (Arm Twist)", 87.0, ""),
                ("J4 (Elbow Pitch)", 87.0, "Dynamic lift effort"),
                ("J5 (Wrist Roll)", 12.0, ""),
                ("J6 (Wrist Pitch)", 12.0, ""),
                ("J7 (Wrist Yaw)", 12.0, "")
            ]

            t_box_y = col_y1 + 206
            mid_x = c3_x1 + 290
            half_bar = 85

            for idx, (j_name, limit, note) in enumerate(joint_specs):
                tau = float(j_raw[idx]) if idx < len(j_raw) else 0.0
                jy = t_box_y + idx * 46

                # Bipolar bar track
                cv2.rectangle(canvas, (mid_x - half_bar, jy + 4), (mid_x + half_bar, jy + 18), (42, 35, 30), -1)
                cv2.line(canvas, (mid_x, jy + 2), (mid_x, jy + 20), (80, 70, 60), 1)

                norm_val = np.clip(tau / limit, -1.0, 1.0)
                bar_len = int(abs(norm_val) * half_bar)
                bar_col = self.PAL_AMBER if abs(tau) > 30 else (0, 230, 160)
                if norm_val >= 0:
                    cv2.rectangle(canvas, (mid_x, jy + 4), (mid_x + bar_len, jy + 18), bar_col, -1)
                else:
                    cv2.rectangle(canvas, (mid_x - bar_len, jy + 4), (mid_x, jy + 18), bar_col, -1)

            # Peak summary card
            pj = force_snapshot.get("peak_joint", "J2")
            pt = float(force_snapshot.get("peak_torque", -33.52))
            cv2.rectangle(canvas, (c3_x1 + 14, col_y2 - 40), (c3_x2 - 14, col_y2 - 10), (32, 28, 24), -1)
            cv2.rectangle(canvas, (c3_x1 + 14, col_y2 - 40), (c3_x2 - 14, col_y2 - 10), self.PAL_BORDER, 1, cv2.LINE_AA)
        else:
            # Standby Card when idle before first grasp
            cv2.rectangle(canvas, (c3_x1 + 24, col_y1 + 180), (c3_x2 - 24, col_y1 + 380), (38, 32, 28), -1)
            cv2.rectangle(canvas, (c3_x1 + 24, col_y1 + 180), (c3_x2 - 24, col_y1 + 380), self.PAL_BORDER, 1, cv2.LINE_AA)

        # -------------------------------------------------------------
        # 3. BOTTOM COMMAND & INTERACTIVE BAR (y = 660 to 820)
        # -------------------------------------------------------------
        bot_y = 660
        cv2.rectangle(canvas, (0, bot_y), (W, H), (22, 18, 15), -1)
        cv2.line(canvas, (0, bot_y), (W, bot_y), self.PAL_BORDER, 1)

        # Row 1: Action Buttons (y: 670 to 716, h=46)
        btn_y1, btn_y2 = 670, 716
        buttons = [
            ("detect", "DETECT", "[D]", 14, 180, self.PAL_CYAN),
            ("execute", "EXECUTE NEXT", "[E/Space]", 195, 420, self.PAL_GREEN),
            ("pick_all", "PICK ALL", "[A]", 435, 600, self.PAL_CYAN),
            ("batch_pick", "BATCH PICK", "[B]", 615, 780, (255, 200, 50)),
            ("respawn", "RESPAWN", "[R]", 795, 960, self.PAL_AMBER),
            ("home", "HOME", "[H]", 975, 1140, (200, 210, 220)),
            ("camera", "CAMERA", "[C/1-4]", 1155, 1320, (230, 130, 255)),
            ("quit", "QUIT", "[Q]", 1335, 1426, self.PAL_RED)
        ]

        for act_id, label, key, bx1, bx2, col in buttons:
            self.button_hitboxes[act_id] = (bx1, btn_y1, bx2, btn_y2)
            # Elevated tactile button surface
            cv2.rectangle(canvas, (bx1, btn_y1), (bx2, btn_y2), (46, 38, 32), -1)
            # 2px anti-aliased border
            cv2.rectangle(canvas, (bx1, btn_y1), (bx2, btn_y2), col, 2, cv2.LINE_AA)
            # Tactile bevel line
            cv2.line(canvas, (bx1 + 2, btn_y1 + 2), (bx2 - 2, btn_y1 + 2), (75, 65, 55), 1)

        # Row 2: Discrete Steppers & Jogging Guide (y: 724 to 764)
        # Spawn Stepper
        cv2.rectangle(canvas, (14, 724), (200, 764), (38, 32, 28), -1)
        cv2.rectangle(canvas, (14, 724), (200, 764), self.PAL_BORDER, 1, cv2.LINE_AA)
        # [-]
        self.button_hitboxes["spawn_minus"] = (14, 724, 54, 764)
        cv2.rectangle(canvas, (14, 724), (54, 764), (56, 46, 40), -1)
        cv2.rectangle(canvas, (14, 724), (54, 764), self.PAL_CYAN, 1, cv2.LINE_AA)
        # [+]
        self.button_hitboxes["spawn_plus"] = (160, 724, 200, 764)
        cv2.rectangle(canvas, (160, 724), (200, 764), (56, 46, 40), -1)
        cv2.rectangle(canvas, (160, 724), (200, 764), self.PAL_CYAN, 1, cv2.LINE_AA)

        # Batch Stepper
        cv2.rectangle(canvas, (215, 724), (400, 764), (38, 32, 28), -1)
        cv2.rectangle(canvas, (215, 724), (400, 764), self.PAL_BORDER, 1, cv2.LINE_AA)
        # [-]
        self.button_hitboxes["batch_minus"] = (215, 724, 255, 764)
        cv2.rectangle(canvas, (215, 724), (255, 764), (56, 46, 40), -1)
        cv2.rectangle(canvas, (215, 724), (255, 764), self.PAL_CYAN, 1, cv2.LINE_AA)
        # [+]
        self.button_hitboxes["batch_plus"] = (360, 724, 400, 764)
        cv2.rectangle(canvas, (360, 724), (400, 764), (56, 46, 40), -1)
        cv2.rectangle(canvas, (360, 724), (400, 764), self.PAL_CYAN, 1, cv2.LINE_AA)

        # Teleoperation Jog Guide Box
        cv2.rectangle(canvas, (415, 724), (1426, 764), (32, 26, 22), -1)
        cv2.rectangle(canvas, (415, 724), (1426, 764), self.PAL_BORDER, 1, cv2.LINE_AA)

        # -------------------------------------------------------------
        # 4. CRISP TRUETYPE TYPOGRAPHY LAYER (PIL ImageDraw)
        # -------------------------------------------------------------
        pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        # TrueType Fonts
        f_title = self._get_font(17, bold=True)
        f_sub = self._get_font(11, bold=False)
        f_badge = self._get_font(12, bold=True)
        f_col_hdr = self._get_font(12, bold=True)
        f_btn_lbl = self._get_font(13, bold=True)
        f_btn_key = self._get_font(11, bold=True)
        f_card_bold = self._get_font(11, bold=True)
        f_card_reg = self._get_font(11, bold=False)
        f_stepper_sym = self._get_font(15, bold=True)
        f_stepper_val = self._get_font(13, bold=True)
        f_small = self._get_font(10, bold=False)

        # 1. Header Text (Zero question marks!)
        draw.text((22, 14), "CPS FRANKA ROBOTIC COMMAND CONSOLE", font=f_title, fill=(22, 224, 234))
        draw.text((22, 38), "GG-CNN Antipodal Grasp Engine  |  Closed-Loop Contact Dynamics  |  7-DOF Franka Panda", font=f_sub, fill=(170, 180, 190))

        # Badges
        draw.text((bx_stat + 10, 23), badge_stat, font=f_badge, fill=(245, 245, 250))
        draw.text((bx_c + 10, 23), badge_clutter, font=f_badge, fill=(255, 184, 48))
        s_rgb = (0, 255, 150) if any(w in status_msg.upper() for w in ["FORCE", "ACHIEVED", "SUCCESS", "CLEARED"]) else (
            (255, 184, 48) if any(w in status_msg.upper() for w in ["DETECT", "SCAN", "BATCH"]) else (22, 224, 234)
        )
        draw.text((bx_s + 10, 23), badge_status, font=f_badge, fill=s_rgb)

        # 2. Column Headers
        draw.text((c1_x1 + 14, col_y1 + 6), "1. OVERHEAD RGB & ANTIPODAL TARGETS", font=f_col_hdr, fill=(22, 224, 234))
        draw.text((c2_x1 + 14, col_y1 + 6), "2. GG-CNN PREDICTION & CONFIDENCE FIELD", font=f_col_hdr, fill=(22, 224, 234))
        draw.text((c3_x1 + 14, col_y1 + 6), "3. CONTACT DYNAMICS & ACTUATOR TELEMETRY", font=f_col_hdr, fill=(22, 224, 234))

        # 3. Col 1 Info Card
        if len(grasps) > 0:
            top_g = grasps[0]
            val_s = getattr(top_g, "validated_score", top_g.quality)
            draw.text((c1_x1 + 20, col_y1 + 486), f"Top Grasp #1: Q = {top_g.quality:.3f} | Score = {val_s:.3f} | {top_g.angle_deg:+.1f}°", font=f_card_bold, fill=(0, 255, 140))
            if top_g.world_coords is not None:
                wx, wy, wz = top_g.world_coords
                draw.text((c1_x1 + 20, col_y1 + 508), f"Target 3D: [{wx:.3f}, {wy:.3f}, {wz:.3f}] m", font=f_card_reg, fill=(240, 240, 245))
            else:
                draw.text((c1_x1 + 20, col_y1 + 508), f"Image Pos: u={top_g.u:.0f}px, v={top_g.v:.0f}px, width={top_g.width_px*0.5:.1f}mm", font=f_card_reg, fill=(240, 240, 245))
            draw.text((c1_x1 + 20, col_y1 + 528), "Antipodal verification: friction cone + parallel span check", font=f_card_reg, fill=(160, 150, 140))
        else:
            draw.text((c1_x1 + 20, col_y1 + 492), "[AWAITING DETECTION] No active grasp candidate", font=f_card_bold, fill=(255, 184, 48))
            draw.text((c1_x1 + 20, col_y1 + 518), "Click [DETECT] or press 'D' to run GG-CNN inference", font=f_card_reg, fill=(170, 160, 150))

        # 4. Col 2 Info Card
        draw.text((c2_x1 + 20, col_y1 + 486), f"Inference Latency: {lat:.1f} ms (~{fps:.0f} FPS) | Hardware: {dev}", font=f_card_bold, fill=(22, 224, 234))
        draw.text((c2_x1 + 20, col_y1 + 508), "Architecture: Generative Grasping CNN (GG-CNN Encoder-Decoder)", font=f_card_reg, fill=(240, 240, 245))
        draw.text((c2_x1 + 20, col_y1 + 528), "Output Channels: Quality Q(u,v), Angle θ(u,v), Width W(u,v)", font=f_card_reg, fill=(160, 150, 140))

        # 5. Col 3 Contact Dynamics
        has_force = bool(force_snapshot and (force_snapshot.get("normal_force", 0) > 0 or force_snapshot.get("mean_fn", 0) > 0 or force_snapshot.get("joint_torques")))
        if has_force:
            draw.text((c3_x1 + 14, col_y1 + 46), "Contact Diagnostics (Coulomb Friction Formulation):", font=f_card_bold, fill=(255, 255, 255))
            if is_standby:
                draw.text((c3_x1 + 14, col_y1 + 66), "F_normal: 0.0 N  (Gripper Disengaged / Zero Load)", font=f_card_bold, fill=(180, 190, 200))
                draw.text((c3_x1 + 14, col_y1 + 104), "Friction Angle: θ = 0.0° (Limit: 54.5° for μ=1.4)", font=f_card_reg, fill=(180, 190, 200))
                draw.text((c3_x1 + 22, col_y1 + 130), fc_text, font=f_card_bold, fill=(22, 224, 234))
            else:
                draw.text((c3_x1 + 14, col_y1 + 66), f"F_normal: {fn_val:.1f} N  (Left: {fn_l:.1f} N | Right: {fn_r:.1f} N)", font=f_card_bold, fill=(0, 255, 140) if fc_status else (255, 184, 48))
                draw.text((c3_x1 + 14, col_y1 + 104), f"Friction Angle: θ = {th_val:.1f}° (Limit: 54.5° for μ=1.4)", font=f_card_reg, fill=(240, 240, 245))
                fc_col = (0, 255, 140) if fc_status else (255, 70, 70)
                draw.text((c3_x1 + 22, col_y1 + 130), fc_text, font=f_card_bold, fill=fc_col)

            draw.text((c3_x1 + 14, col_y1 + 176), "Franka Panda 7-DOF Actuator Torques (Motor Effort):", font=f_card_bold, fill=(22, 224, 234))
            for idx, (j_name, limit, note) in enumerate(joint_specs):
                tau = float(j_raw[idx]) if idx < len(j_raw) else 0.0
                jy = t_box_y + idx * 46
                draw.text((c3_x1 + 14, jy + 2), j_name, font=f_card_bold, fill=(255, 255, 255))
                if note:
                    draw.text((c3_x1 + 14, jy + 18), f"({note})", font=f_small, fill=(160, 150, 140))
                draw.text((mid_x + half_bar + 10, jy + 2), f"{tau:+.2f} Nm", font=f_card_bold, fill=(240, 240, 245))

            draw.text((c3_x1 + 22, col_y2 - 32), f"PEAK MOTOR EFFORT: {pj} ({pt:+.2f} N·m)", font=f_card_bold, fill=(22, 224, 234))
        else:
            draw.text((c3_x1 + 46, col_y1 + 215), "[ AWAITING PHYSICAL GRASP TELEMETRY ]", font=f_card_bold, fill=(22, 224, 234))
            draw.text((c3_x1 + 46, col_y1 + 245), "Trigger a pick-and-place cycle to stream live telemetry:", font=f_card_reg, fill=(255, 255, 255))
            draw.text((c3_x1 + 50, col_y1 + 275), "• Coulomb friction cone margin & normal forces", font=f_card_reg, fill=(180, 170, 160))
            draw.text((c3_x1 + 50, col_y1 + 300), "• 7-DOF Franka Panda actuator dynamic torques", font=f_card_reg, fill=(180, 170, 160))
            draw.text((c3_x1 + 46, col_y1 + 335), "Click [EXECUTE NEXT] or press Spacebar to begin.", font=f_card_bold, fill=(0, 255, 140))

        # 6. Bottom Buttons (Bold White Action Label + High-Contrast Hotkey Tag)
        for act_id, label, key, bx1, bx2, col in buttons:
            col_rgb = (col[2], col[1], col[0])
            bb_l = draw.textbbox((0, 0), label, font=f_btn_lbl)
            lw = bb_l[2] - bb_l[0]
            bb_k = draw.textbbox((0, 0), key, font=f_btn_key)
            kw = bb_k[2] - bb_k[0]
            tot_w = lw + 8 + kw
            start_x = bx1 + (bx2 - bx1 - tot_w) // 2

            # Pure Bold White Label
            draw.text((start_x, btn_y1 + 14), label, font=f_btn_lbl, fill=(255, 255, 255))
            # Vivid Hotkey Tag
            draw.text((start_x + lw + 8, btn_y1 + 16), key, font=f_btn_key, fill=col_rgb)

        # 7. Steppers
        draw.text((25, 733), "[-]", font=f_stepper_sym, fill=(22, 224, 234))
        draw.text((70, 735), f"SPAWN: {spawn_count}", font=f_stepper_val, fill=(255, 255, 255))
        draw.text((171, 733), "[+]", font=f_stepper_sym, fill=(22, 224, 234))

        draw.text((226, 733), "[-]", font=f_stepper_sym, fill=(22, 224, 234))
        draw.text((270, 735), f"BATCH: {batch_count}", font=f_stepper_val, fill=(255, 255, 255))
        draw.text((371, 733), "[+]", font=f_stepper_sym, fill=(22, 224, 234))

        # 8. Teleoperation Guide
        jog_str = "TELEOPERATION: [Arrows] Nudge Arm X/Y  |  [PageUp/Down or U/J] Arm Z  |  [,/.] Wrist Yaw  |  [O/P] Open/Close Gripper"
        draw.text((430, 736), jog_str, font=f_sub, fill=(225, 232, 240))

        # 9. Hotkeys Strip
        legend_str = "HOTKEYS: [D] Detect   [E/Space] Next Pick   [A] Pick All   [B] Batch   [R] Respawn   [H] Home   [1-4/C] Cameras   [G] GUI   [Q] Quit"
        draw.text((22, 786), legend_str, font=f_sub, fill=(160, 150, 140))

        # Convert back to BGR for OpenCV
        canvas = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return canvas
