"""
Prototype test for the Industrial Command Console layout and components.
"""
import os
import cv2
import numpy as np

def build_prototype():
    W, H = 1440, 820
    canvas = np.full((H, W, 3), (26, 21, 18), dtype=np.uint8)

    # Palette (BGR)
    BG_DARK = (26, 21, 18)
    BG_PANEL = (34, 28, 24)
    BORDER = (58, 48, 42)
    CYAN = (234, 224, 22)
    GREEN = (150, 255, 0)
    AMBER = (48, 184, 255)
    RED = (96, 64, 255)
    WHITE = (240, 236, 232)
    MUTED = (165, 149, 138)

    # 1. TOP HEADER (h = 60)
    top_h = 60
    cv2.rectangle(canvas, (0, 0), (W, top_h), (32, 26, 22), -1)
    cv2.line(canvas, (0, top_h), (W, top_h), BORDER, 1)

    cv2.putText(canvas, "✦ CPS FRANKA ROBOTIC COMMAND CONSOLE ✦", (20, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, CYAN, 2, cv2.LINE_AA)
    cv2.putText(canvas, "GG-CNN Antipodal Grasp Engine | Closed-Loop Contact Dynamics | 7-DOF Franka Panda",
                (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.38, MUTED, 1, cv2.LINE_AA)

    # Right side badges
    badge_fc = "FORCE CLOSURE: [ACHIEVED]"
    (bw1, _), _ = cv2.getTextSize(badge_fc, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
    bx1 = W - bw1 - 250
    cv2.rectangle(canvas, (bx1 - 8, 14), (bx1 + bw1 + 8, 46), (36, 45, 30), -1)
    cv2.rectangle(canvas, (bx1 - 8, 14), (bx1 + bw1 + 8, 46), GREEN, 1, cv2.LINE_AA)
    cv2.putText(canvas, badge_fc, (bx1, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.44, GREEN, 1, cv2.LINE_AA)

    badge_stat = "SUCCESS: 5/5 (100%)"
    (bw2, _), _ = cv2.getTextSize(badge_stat, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
    bx2 = W - bw2 - 25
    cv2.rectangle(canvas, (bx2 - 8, 14), (bx2 + bw2 + 8, 46), (40, 34, 28), -1)
    cv2.rectangle(canvas, (bx2 - 8, 14), (bx2 + bw2 + 8, 46), CYAN, 1, cv2.LINE_AA)
    cv2.putText(canvas, badge_stat, (bx2, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.44, WHITE, 1, cv2.LINE_AA)

    # 2. MAIN AREA: 3 COLUMNS (y = 66 to 650)
    col_y1, col_y2 = 68, 650
    col_h = col_y2 - col_y1

    # Col 1: RGB Perception (x: 14 to 468, W = 454)
    c1_x1, c1_x2 = 14, 468
    cv2.rectangle(canvas, (c1_x1, col_y1), (c1_x2, col_y2), BG_PANEL, -1)
    cv2.rectangle(canvas, (c1_x1, col_y1), (c1_x2, col_y2), BORDER, 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (c1_x1, col_y1), (c1_x2, col_y1 + 28), (42, 34, 30), -1)
    cv2.putText(canvas, "1. OVERHEAD RGB & ANTIPODAL TARGETS", (c1_x1 + 12, col_y1 + 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, CYAN, 1, cv2.LINE_AA)

    # Mock RGB view inside Col 1 (430 x 430)
    rgb_mock = np.full((430, 430, 3), 40, dtype=np.uint8)
    cv2.circle(rgb_mock, (215, 215), 50, (80, 80, 180), -1)
    cv2.putText(rgb_mock, "Target Object", (160, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    canvas[col_y1 + 36:col_y1 + 36 + 430, c1_x1 + 12:c1_x1 + 12 + 430] = rgb_mock

    # Col 1 Bottom Info Card
    cv2.rectangle(canvas, (c1_x1 + 12, col_y1 + 476), (c1_x2 - 12, col_y2 - 10), (28, 23, 20), -1)
    cv2.rectangle(canvas, (c1_x1 + 12, col_y1 + 476), (c1_x2 - 12, col_y2 - 10), BORDER, 1, cv2.LINE_AA)
    cv2.putText(canvas, "Top Grasp #1: Q = 0.84 | Score = 0.88 | Angle = +12.4 deg",
                (c1_x1 + 20, col_y1 + 498), cv2.FONT_HERSHEY_SIMPLEX, 0.38, GREEN, 1, cv2.LINE_AA)
    cv2.putText(canvas, "World 3D Target: X=0.482m, Y=0.015m, Z=0.012m",
                (c1_x1 + 20, col_y1 + 520), cv2.FONT_HERSHEY_SIMPLEX, 0.36, WHITE, 1, cv2.LINE_AA)
    cv2.putText(canvas, "Status: antipodal validation passed (clearance + kinematics)",
                (c1_x1 + 20, col_y1 + 540), cv2.FONT_HERSHEY_SIMPLEX, 0.34, MUTED, 1, cv2.LINE_AA)

    # Col 2: GG-CNN Heatmap (x: 480 to 934, W = 454)
    c2_x1, c2_x2 = 480, 934
    cv2.rectangle(canvas, (c2_x1, col_y1), (c2_x2, col_y2), BG_PANEL, -1)
    cv2.rectangle(canvas, (c2_x1, col_y1), (c2_x2, col_y2), BORDER, 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (c2_x1, col_y1), (c2_x2, col_y1 + 28), (42, 34, 30), -1)
    cv2.putText(canvas, "2. GG-CNN PREDICTION & CONFIDENCE FIELD", (c2_x1 + 12, col_y1 + 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, CYAN, 1, cv2.LINE_AA)

    # Mock Heatmap inside Col 2
    q_mock = np.zeros((430, 430), dtype=np.uint8)
    cv2.circle(q_mock, (215, 215), 60, 255, -1)
    q_mock = cv2.GaussianBlur(q_mock, (61, 61), 0)
    q_col = cv2.applyColorMap(q_mock, cv2.COLORMAP_INFERNO)
    canvas[col_y1 + 36:col_y1 + 36 + 430, c2_x1 + 12:c2_x1 + 12 + 430] = q_col

    # Col 2 Bottom Info Card
    cv2.rectangle(canvas, (c2_x1 + 12, col_y1 + 476), (c2_x2 - 12, col_y2 - 10), (28, 23, 20), -1)
    cv2.rectangle(canvas, (c2_x1 + 12, col_y1 + 476), (c2_x2 - 12, col_y2 - 10), BORDER, 1, cv2.LINE_AA)
    cv2.putText(canvas, "Inference Latency: 24.1 ms (~41 FPS) | Hardware: CUDA",
                (c2_x1 + 20, col_y1 + 498), cv2.FONT_HERSHEY_SIMPLEX, 0.38, CYAN, 1, cv2.LINE_AA)
    cv2.putText(canvas, "Architecture: Generative Grasping CNN (GG-CNN Encoder-Decoder)",
                (c2_x1 + 20, col_y1 + 520), cv2.FONT_HERSHEY_SIMPLEX, 0.34, WHITE, 1, cv2.LINE_AA)
    cv2.putText(canvas, "Output Channels: Quality Q(u,v), Angle theta(u,v), Width W(u,v)",
                (c2_x1 + 20, col_y1 + 540), cv2.FONT_HERSHEY_SIMPLEX, 0.34, MUTED, 1, cv2.LINE_AA)

    # Col 3: Cyber-Physical Contact Dynamics & Actuators (x: 946 to 1426, W = 480)
    c3_x1, c3_x2 = 946, 1426
    cv2.rectangle(canvas, (c3_x1, col_y1), (c3_x2, col_y2), BG_PANEL, -1)
    cv2.rectangle(canvas, (c3_x1, col_y1), (c3_x2, col_y2), BORDER, 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (c3_x1, col_y1), (c3_x2, col_y1 + 28), (42, 34, 30), -1)
    cv2.putText(canvas, "3. CONTACT DYNAMICS & ACTUATOR TELEMETRY", (c3_x1 + 12, col_y1 + 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, CYAN, 1, cv2.LINE_AA)

    # Contact Diagnostics (Normal force, shear force, friction angle, margin)
    cv2.putText(canvas, "Contact Diagnostics (Coulomb Friction Formulation):",
                (c3_x1 + 14, col_y1 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.38, WHITE, 1, cv2.LINE_AA)

    # Normal Force Bar
    fn_val = 139.2
    fn_max = 180.0
    bar_w = 340
    cv2.putText(canvas, f"F_normal: {fn_val:.1f} N (Target: 120 N)",
                (c3_x1 + 14, col_y1 + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.36, GREEN, 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (c3_x1 + 14, col_y1 + 78), (c3_x1 + 14 + bar_w, col_y1 + 92), (45, 38, 32), -1)
    fill_w = int(bar_w * min(1.0, fn_val / fn_max))
    cv2.rectangle(canvas, (c3_x1 + 14, col_y1 + 78), (c3_x1 + 14 + fill_w, col_y1 + 92), (0, 220, 140), -1)
    # Target line at 120 N
    t_x = c3_x1 + 14 + int(bar_w * (120.0 / fn_max))
    cv2.line(canvas, (t_x, col_y1 + 76), (t_x, col_y1 + 94), (0, 255, 255), 2)

    # Friction Angle & Margin
    cv2.putText(canvas, "Friction Angle: theta = 16.4 deg (Cone Limit: 54.5 deg for mu=1.4)",
                (c3_x1 + 14, col_y1 + 114), cv2.FONT_HERSHEY_SIMPLEX, 0.36, WHITE, 1, cv2.LINE_AA)
    # Margin pill
    cv2.rectangle(canvas, (c3_x1 + 14, col_y1 + 124), (c3_x2 - 14, col_y1 + 152), (32, 45, 28), -1)
    cv2.rectangle(canvas, (c3_x1 + 14, col_y1 + 124), (c3_x2 - 14, col_y1 + 152), GREEN, 1, cv2.LINE_AA)
    cv2.putText(canvas, "VERDICT: [FORCE CLOSURE ACHIEVED] (Safety Margin: 79.0%)",
                (c3_x1 + 24, col_y1 + 143), cv2.FONT_HERSHEY_SIMPLEX, 0.38, GREEN, 1, cv2.LINE_AA)

    # Franka Panda 7-DOF Actuator Joint Torques
    cv2.line(canvas, (c3_x1 + 14, col_y1 + 168), (c3_x2 - 14, col_y1 + 168), BORDER, 1)
    cv2.putText(canvas, "Franka Panda 7-DOF Actuator Torques (Motor Effort):",
                (c3_x1 + 14, col_y1 + 190), cv2.FONT_HERSHEY_SIMPLEX, 0.38, CYAN, 1, cv2.LINE_AA)

    joints = [
        ("J1 (Base Yaw)", -0.02, 87.0, ""),
        ("J2 (Shoulder)", -33.52, 87.0, "Countering gravity"),
        ("J3 (Arm Twist)", +0.17, 87.0, ""),
        ("J4 (Elbow Pitch)", +20.12, 87.0, "Dynamic lift effort"),
        ("J5 (Wrist Roll)", +0.99, 12.0, ""),
        ("J6 (Wrist Pitch)", +1.71, 12.0, ""),
        ("J7 (Wrist Yaw)", +0.02, 12.0, "")
    ]

    t_box_y = col_y1 + 206
    mid_x = c3_x1 + 290
    half_bar = 85

    for idx, (name, tau, limit, note) in enumerate(joints):
        jy = t_box_y + idx * 46
        cv2.putText(canvas, name, (c3_x1 + 14, jy + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, WHITE, 1, cv2.LINE_AA)
        if note:
            cv2.putText(canvas, f"({note})", (c3_x1 + 14, jy + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.30, MUTED, 1, cv2.LINE_AA)

        # Bipolar bar track: [mid_x - half_bar, mid_x + half_bar]
        cv2.rectangle(canvas, (mid_x - half_bar, jy + 4), (mid_x + half_bar, jy + 18), (42, 35, 30), -1)
        cv2.line(canvas, (mid_x, jy + 2), (mid_x, jy + 20), (80, 70, 60), 1)

        # Active bar fill
        norm_val = np.clip(tau / limit, -1.0, 1.0)
        bar_len = int(abs(norm_val) * half_bar)
        bar_col = AMBER if abs(tau) > 30 else (0, 230, 160)
        if norm_val >= 0:
            cv2.rectangle(canvas, (mid_x, jy + 4), (mid_x + bar_len, jy + 18), bar_col, -1)
        else:
            cv2.rectangle(canvas, (mid_x - bar_len, jy + 4), (mid_x, jy + 18), bar_col, -1)

        val_str = f"{tau:+.2f} Nm"
        cv2.putText(canvas, val_str, (mid_x + half_bar + 10, jy + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, WHITE, 1, cv2.LINE_AA)

    # Peak Effort summary
    cv2.rectangle(canvas, (c3_x1 + 14, col_y2 - 40), (c3_x2 - 14, col_y2 - 10), (32, 28, 24), -1)
    cv2.rectangle(canvas, (c3_x1 + 14, col_y2 - 40), (c3_x2 - 14, col_y2 - 10), BORDER, 1, cv2.LINE_AA)
    cv2.putText(canvas, "PEAK MOTOR EFFORT: J2 Shoulder (-33.52 Nm) | J4 Elbow (+20.12 Nm)",
                (c3_x1 + 22, col_y2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.34, CYAN, 1, cv2.LINE_AA)

    # 3. BOTTOM COMMAND & INTERACTIVE BAR (y = 660 to 820)
    bot_y = 660
    cv2.rectangle(canvas, (0, bot_y), (W, H), (22, 18, 15), -1)
    cv2.line(canvas, (0, bot_y), (W, bot_y), BORDER, 1)

    # Row 1: Action Buttons (y: 672 to 714)
    btn_y1, btn_y2 = 672, 714
    buttons = [
        ("DETECT [D]", 14, 180, CYAN),
        ("EXECUTE NEXT [E]", 195, 420, GREEN),
        ("PICK ALL [A]", 435, 600, CYAN),
        ("BATCH PICK [B]", 615, 780, (255, 200, 50)),
        ("RESPAWN [R]", 795, 960, AMBER),
        ("HOME [H]", 975, 1140, (180, 190, 200)),
        ("CAMERA [C/1-4]", 1155, 1320, (230, 130, 255)),
        ("QUIT [Q]", 1335, 1426, RED)
    ]
    for label, bx1, bx2, col in buttons:
        cv2.rectangle(canvas, (bx1, btn_y1), (bx2, btn_y2), (36, 30, 26), -1)
        cv2.rectangle(canvas, (bx1, btn_y1), (bx2, btn_y2), col, 1, cv2.LINE_AA)
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        tx = bx1 + (bx2 - bx1 - tw) // 2
        cv2.putText(canvas, label, (tx, btn_y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)

    # Row 2: Discrete Steppers & Jogging Guide (y: 724 to 764)
    # Spawn Stepper
    cv2.rectangle(canvas, (14, 724), (200, 764), (34, 28, 24), -1)
    cv2.rectangle(canvas, (14, 724), (200, 764), BORDER, 1, cv2.LINE_AA)
    # [-]
    cv2.rectangle(canvas, (14, 724), (54, 764), (45, 36, 30), -1)
    cv2.putText(canvas, "[-]", (24, 748), cv2.FONT_HERSHEY_SIMPLEX, 0.44, CYAN, 1, cv2.LINE_AA)
    # Text
    cv2.putText(canvas, "SPAWN: 3", (68, 748), cv2.FONT_HERSHEY_SIMPLEX, 0.40, WHITE, 1, cv2.LINE_AA)
    # [+]
    cv2.rectangle(canvas, (160, 724), (200, 764), (45, 36, 30), -1)
    cv2.putText(canvas, "[+]", (168, 748), cv2.FONT_HERSHEY_SIMPLEX, 0.44, CYAN, 1, cv2.LINE_AA)

    # Batch Stepper
    cv2.rectangle(canvas, (215, 724), (400, 764), (34, 28, 24), -1)
    cv2.rectangle(canvas, (215, 724), (400, 764), BORDER, 1, cv2.LINE_AA)
    # [-]
    cv2.rectangle(canvas, (215, 724), (255, 764), (45, 36, 30), -1)
    cv2.putText(canvas, "[-]", (225, 748), cv2.FONT_HERSHEY_SIMPLEX, 0.44, CYAN, 1, cv2.LINE_AA)
    # Text
    cv2.putText(canvas, "BATCH: 2", (270, 748), cv2.FONT_HERSHEY_SIMPLEX, 0.40, WHITE, 1, cv2.LINE_AA)
    # [+]
    cv2.rectangle(canvas, (360, 724), (400, 764), (45, 36, 30), -1)
    cv2.putText(canvas, "[+]", (368, 748), cv2.FONT_HERSHEY_SIMPLEX, 0.44, CYAN, 1, cv2.LINE_AA)

    # Teleoperation Jog Guide
    cv2.rectangle(canvas, (415, 724), (1426, 764), (28, 24, 20), -1)
    cv2.rectangle(canvas, (415, 724), (1426, 764), BORDER, 1, cv2.LINE_AA)
    jog_str = "TELEOPERATION: [Arrows] Nudge Arm X/Y | [PageUp/Down or U/J] Arm Z | [,/.] Wrist Yaw | [O/P] Open/Close Gripper"
    cv2.putText(canvas, jog_str, (430, 748), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (210, 220, 230), 1, cv2.LINE_AA)

    # Row 3: Hotkey Legend Strip (y: 776 to 810)
    legend_str = "HOTKEYS: [D] Detect  [E/Space] Next Pick  [A] Pick All  [B] Batch  [R] Respawn  [H] Home  [1-4/C] Cameras  [Arrows/U/J] Jog  [O/P] Gripper  [Q] Quit"
    cv2.putText(canvas, legend_str, (20, 796), cv2.FONT_HERSHEY_SIMPLEX, 0.35, MUTED, 1, cv2.LINE_AA)

    cv2.imwrite("outputs/test_console.png", canvas)
    print("Command Console Prototype generated at outputs/test_console.png")

if __name__ == "__main__":
    build_prototype()
