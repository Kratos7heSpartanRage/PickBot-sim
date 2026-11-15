"""
=============================================================================
  CYBER PHYSICAL SYSTEMS: FORCE CLOSURE & JOINT TORQUE MONITOR
  Real-Time Contact Mechanics & Actuator Telemetry for Robotic Manipulation
=============================================================================
Theory:
  - Antipodal Grasps achieve Force Closure when opposing normal contact forces
    and tangential friction forces satisfy Coulomb's friction cone:
      |F_tangential| <= mu * F_normal  ==>  theta = arctan(F_t / F_n) <= arctan(mu)
  - Joint Torques (Inverse Dynamics):
      tau = M(q)*q_ddot + C(q, q_dot)*q_dot + g(q) + J^T(q)*F_ext
=============================================================================
"""
import time
import os
import numpy as np
import pybullet as p
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class ForceClosureMonitor:
    """
    Real-time Cyber-Physical contact mechanics and actuator torque analyzer.
    Tracks gripper normal forces, friction cone alignment, force closure,
    and 7-DOF arm joint torques throughout manipulation phases.
    """

    JOINT_NAMES = [
        "J1 (Base Yaw)",
        "J2 (Shoulder)",
        "J3 (Arm Twist)",
        "J4 (Elbow Pitch)",
        "J5 (Wrist Roll)",
        "J6 (Wrist Pitch)",
        "J7 (Wrist Yaw)"
    ]

    def __init__(self, robot_id, left_finger_link=9, right_finger_link=10,
                 arm_joints=None, mu=1.4):
        self.robot_id = robot_id
        self.left_finger_link = left_finger_link
        self.right_finger_link = right_finger_link
        self.arm_joints = arm_joints or [0, 1, 2, 3, 4, 5, 6]
        self.mu = float(mu)
        self.theta_cone_deg = float(np.degrees(np.arctan(self.mu)))  # ~54.5 deg

        self.history = []
        self.t_start = None

    def reset(self):
        """Clears telemetry buffer for a fresh pick-and-place trial."""
        self.history.clear()
        self.t_start = time.perf_counter()

    def sample(self, phase_name="Unknown"):
        """
        Samples contact mechanics and actuator torques from PyBullet.
        Returns the sampled telemetry snapshot dict.
        """
        if self.t_start is None:
            self.t_start = time.perf_counter()
        t_elapsed = time.perf_counter() - self.t_start

        # 1. Query contact points on gripper finger pads
        left_fn = 0.0
        left_ft_sq = 0.0
        right_fn = 0.0
        right_ft_sq = 0.0
        left_pos = None
        right_pos = None

        try:
            contacts = p.getContactPoints(bodyA=self.robot_id)
            for c in contacts:
                link_a = c[3]
                body_b = c[2]
                # Filter out table/plane collisions if necessary (plane is id 0)
                if body_b == 0:
                    continue

                fn = max(0.0, float(c[9]))
                fric1 = float(c[10])
                fric2 = float(c[12])
                ft_sq = fric1**2 + fric2**2

                if link_a == self.left_finger_link:
                    left_fn += fn
                    left_ft_sq += ft_sq
                    left_pos = c[5]
                elif link_a == self.right_finger_link:
                    right_fn += fn
                    right_ft_sq += ft_sq
                    right_pos = c[5]
        except Exception:
            pass

        left_ft = float(np.sqrt(left_ft_sq))
        right_ft = float(np.sqrt(right_ft_sq))

        # Draw 3D force indicators directly in PyBullet scene during grip/lift
        if phase_name in ["Grip", "Lift"]:
            try:
                if left_pos is not None and left_fn > 1.0:
                    p.addUserDebugLine(left_pos, [left_pos[0], left_pos[1], left_pos[2] + 0.05],
                                       [0.0, 1.0, 0.4], lineWidth=3.5, lifeTime=2.0)
                    p.addUserDebugText(f"F_N={left_fn:.0f}N", [left_pos[0], left_pos[1], left_pos[2] + 0.06],
                                       [0.0, 1.0, 0.4], textSize=0.85, lifeTime=2.0)
                if right_pos is not None and right_fn > 1.0:
                    p.addUserDebugLine(right_pos, [right_pos[0], right_pos[1], right_pos[2] + 0.05],
                                       [0.0, 1.0, 0.4], lineWidth=3.5, lifeTime=2.0)
                    p.addUserDebugText(f"F_N={right_fn:.0f}N", [right_pos[0], right_pos[1], right_pos[2] + 0.06],
                                       [0.0, 1.0, 0.4], textSize=0.85, lifeTime=2.0)
            except Exception:
                pass

        # Overall resultant contact forces
        mean_fn = (left_fn + right_fn) / 2.0
        mean_ft = (left_ft + right_ft) / 2.0

        # Coulomb Friction Cone Angle
        if mean_fn > 1e-3:
            friction_angle_deg = float(np.degrees(np.arctan(mean_ft / mean_fn)))
            margin_pct = max(0.0, min(100.0, (1.0 - (mean_ft / (self.mu * mean_fn))) * 100.0))
        else:
            friction_angle_deg = 0.0
            margin_pct = 0.0

        # Force Closure condition: antipodal contact on both jaws inside friction cone
        has_two_point_contact = (left_fn > 0.5 and right_fn > 0.5)
        is_force_closure = has_two_point_contact and (friction_angle_deg <= self.theta_cone_deg)

        # 2. Query 7-DOF Arm Joint Torques
        joint_torques = [0.0] * len(self.arm_joints)
        try:
            joint_states = p.getJointStates(self.robot_id, self.arm_joints)
            for i, st in enumerate(joint_states):
                # index 3: appliedJointMotorTorque in N*m
                joint_torques[i] = round(float(st[3]), 2)
        except Exception:
            pass

        # Identify peak carrying torque
        abs_torques = [abs(t) for t in joint_torques]
        peak_idx = int(np.argmax(abs_torques)) if abs_torques else 0
        peak_val = joint_torques[peak_idx] if joint_torques else 0.0
        peak_joint_name = self.JOINT_NAMES[peak_idx] if peak_idx < len(self.JOINT_NAMES) else f"J{peak_idx+1}"

        snapshot = {
            "time": round(t_elapsed, 4),
            "phase": phase_name,
            "left_fn": round(left_fn, 2),
            "left_ft": round(left_ft, 2),
            "right_fn": round(right_fn, 2),
            "right_ft": round(right_ft, 2),
            "mean_fn": round(mean_fn, 2),
            "mean_ft": round(mean_ft, 2),
            "friction_angle_deg": round(friction_angle_deg, 1),
            "friction_limit_deg": round(self.theta_cone_deg, 1),
            "force_closure": is_force_closure,
            "margin_pct": round(margin_pct, 1),
            "joint_torques": joint_torques,
            "peak_joint": peak_joint_name,
            "peak_torque": peak_val
        }

        self.history.append(snapshot)
        return snapshot

    def print_telemetry_report(self, snapshot=None):
        """
        Prints an industrial Cyber-Physical Systems contact mechanics
        and actuator torque telemetry report to the terminal.
        """
        if snapshot is None:
            if not self.history:
                return
            snapshot = self.history[-1]

        fc_status = "FORCE CLOSURE ACHIEVED" if snapshot["force_closure"] else "INCOMPLETE CLOSURE"
        fc_badge = f"\033[92m[{fc_status}]\033[0m" if snapshot["force_closure"] else f"\033[93m[{fc_status}]\033[0m"

        print("\n" + "=" * 76)
        print(f"  CYBER-PHYSICAL CONTACT MECHANICS & ACTUATOR TELEMETRY ({snapshot['phase'].upper()})")
        print("=" * 76)
        print("  Contact Diagnostics (Coulomb Friction Formulation):")
        print(f"    - Left Finger (Link {self.left_finger_link})   : F_normal = {snapshot['left_fn']:5.1f} N | F_tangential = {snapshot['left_ft']:4.1f} N")
        print(f"    - Right Finger (Link {self.right_finger_link})  : F_normal = {snapshot['right_fn']:5.1f} N | F_tangential = {snapshot['right_ft']:4.1f} N")
        print(f"    - Resultant Friction Angle: θ = {snapshot['friction_angle_deg']:4.1f}° (Limit: {snapshot['friction_limit_deg']:.1f}° for μ={self.mu})")
        print(f"    - Force Closure Verdict   : {fc_badge} (Friction Margin: {snapshot['margin_pct']:.1f}%)")
        print()
        print("  Franka Panda 7-DOF Actuator Joint Torques (Motor Effort):")
        for j_idx, (j_name, tau) in enumerate(zip(self.JOINT_NAMES, snapshot["joint_torques"])):
            flag = ""
            if j_idx == 1:
                flag = " <-- Countering gravitational moment"
            elif j_idx == 3:
                flag = " <-- Peak dynamic lifting effort"
            print(f"    - {j_name:<16}: {tau:+6.2f} N·m{flag}")
        print("=" * 76 + "\n")

    def plot_profile(self, save_path="outputs/force_closure_profile.png"):
        """
        Generates and saves a synchronized 2-panel force-torque profile chart:
          - Panel 1: Contact normal force (F_N) and friction force (F_t) across 6 phases.
          - Panel 2: Motor torques across all 7 Franka arm joints.
        """
        if len(self.history) < 2:
            return None

        times = np.array([h["time"] for h in self.history])
        times -= times[0]  # Zero-base timeline
        phases = [h["phase"] for h in self.history]

        fns = np.array([h["mean_fn"] for h in self.history])
        fts = np.array([h["mean_ft"] for h in self.history])
        limits = fns * self.mu

        # Extract 7 joint torque series
        torques_matrix = np.array([h["joint_torques"] for h in self.history])

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
        fig.patch.set_facecolor("#0e1117")
        ax1.set_facecolor("#161b22")
        ax2.set_facecolor("#161b22")

        # Define distinct phase transition bands
        phase_transitions = []
        cur_phase = phases[0]
        cur_start = times[0]
        for t, p_name in zip(times, phases):
            if p_name != cur_phase:
                phase_transitions.append((cur_phase, cur_start, t))
                cur_phase = p_name
                cur_start = t
        phase_transitions.append((cur_phase, cur_start, times[-1]))

        # Panel 1: Contact Mechanics & Friction Cone
        ax1.plot(times, fns, label=r"Normal Clamping Force $F_N$ (N)", color="#00ffcc", linewidth=2.4)
        ax1.plot(times, fts, label=r"Tangential Friction Force $F_t$ (N)", color="#ff9900", linewidth=2.0)
        ax1.plot(times, limits, label=r"Coulomb Slip Boundary $\mu F_N$ ($\mu=1.4$)",
                 color="#ff3366", linestyle="--", linewidth=1.5, alpha=0.85)

        ax1.set_title("Cyber-Physical Contact Dynamics & Coulomb Force Closure",
                      color="#f0f6fc", fontsize=12, fontweight="bold", pad=10)
        ax1.set_ylabel("Contact Force (N)", color="#f0f6fc", fontsize=10)
        ax1.tick_params(colors="#8b949e")
        ax1.grid(True, linestyle=":", alpha=0.3, color="#8b949e")
        ax1.legend(loc="upper left", facecolor="#161b22", edgecolor="#30363d", labelcolor="#f0f6fc", fontsize=9)

        # Draw vertical phase bands on Panel 1
        colors_band = ["#1f242c", "#282e38"]
        for idx, (p_name, t_s, t_e) in enumerate(phase_transitions):
            bg = colors_band[idx % 2]
            ax1.axvspan(t_s, t_e, color=bg, alpha=0.5, zorder=0)
            ax2.axvspan(t_s, t_e, color=bg, alpha=0.5, zorder=0)
            mid_t = (t_s + t_e) / 2.0
            ax1.text(mid_t, ax1.get_ylim()[1] * 0.90, p_name, color="#8b949e",
                     ha="center", fontsize=8, fontweight="bold")

        # Panel 2: 7-DOF Joint Torques
        joint_colors = ["#4D96FF", "#6BCB77", "#FFD93D", "#FF6B6B", "#C084FC", "#38BDF8", "#F472B6"]
        for j_i in range(min(7, torques_matrix.shape[1])):
            lbl = self.JOINT_NAMES[j_i]
            lw = 2.4 if j_i in (1, 3) else 1.3  # Highlight shoulder and elbow
            ax2.plot(times, torques_matrix[:, j_i], label=lbl, color=joint_colors[j_i], linewidth=lw)

        ax2.set_title("Franka Emika Panda 7-DOF Actuator Joint Torques",
                      color="#f0f6fc", fontsize=12, fontweight="bold", pad=10)
        ax2.set_xlabel("Pick-and-Place Cycle Time (seconds)", color="#f0f6fc", fontsize=10)
        ax2.set_ylabel("Applied Motor Torque (N·m)", color="#f0f6fc", fontsize=10)
        ax2.tick_params(colors="#8b949e")
        ax2.grid(True, linestyle=":", alpha=0.3, color="#8b949e")
        ax2.legend(loc="lower left", facecolor="#161b22", edgecolor="#30363d",
                   labelcolor="#f0f6fc", fontsize=8, ncol=4)

        plt.tight_layout()
        plt.savefig(save_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close()
        return save_path

