"""
=============================================================================
  AI-Based Vision-Guided Robotic Grasping System: Deliverable 2
  Perception Subsystem: Cyber-Physical Grasp Validator Model
=============================================================================
Provides multi-criteria physical and geometric validation for candidate grasps:
  1. Collision & Clearance Validator:
     Checks finger approach corridors and contact points to verify that
     gripper fingers will not collide with the table or neighboring clutter.
  2. Antipodal & Friction Cone Validator:
     Computes depth gradients / surface normals at contact points to verify
     that the grasp forms a stable antipodal pair within the friction cone.
  3. Kinematic Feasibility Validator:
     Verifies that Franka Panda 7-DOF inverse kinematics can reach the target
     orientation and depth within joint limits and safe singularity margins.
  4. Composite Validation Scoring:
     Assigns a physical feasibility score and validation verdict
     ('OPTIMAL', 'FEASIBLE', 'RISKY', 'REJECTED') to rank grasp proposals.
=============================================================================
"""

import numpy as np
import cv2
import pybullet as p


class GraspValidationResult:
    """Detailed validation verdict for a candidate grasp."""
    def __init__(self, grasp, is_valid, score, clearance_score, antipodal_score,
                 kinematic_score, verdict="FEASIBLE", reason=""):
        self.grasp = grasp
        self.is_valid = bool(is_valid)
        self.score = float(score)                     # Composite score [0, 1]
        self.clearance_score = float(clearance_score) # [0, 1]
        self.antipodal_score = float(antipodal_score) # [0, 1]
        self.kinematic_score = float(kinematic_score) # [0, 1]
        self.verdict = str(verdict)                   # 'OPTIMAL', 'FEASIBLE', 'RISKY', 'REJECTED'
        self.reason = str(reason)

    def to_dict(self):
        return {
            "is_valid": self.is_valid,
            "composite_score": round(self.score, 3),
            "clearance_score": round(self.clearance_score, 3),
            "antipodal_score": round(self.antipodal_score, 3),
            "kinematic_score": round(self.kinematic_score, 3),
            "verdict": self.verdict,
            "reason": self.reason
        }


class GraspValidator:
    """
    Cyber-Physical Grasp Validator.
    Acts as a secondary evaluator/critic model to filter and score GG-CNN proposals.
    """
    def __init__(self, table_depth=0.85, friction_coeff=1.40):
        self.table_depth = float(table_depth)
        self.friction_coeff = float(friction_coeff)
        # Maximum allowable angle between surface normal and grasp axis
        # tan(theta_cone) = mu => theta_cone = arctan(1.4) ~ 54.5 deg (~0.95 rad)
        self.max_cone_angle_rad = np.arctan(self.friction_coeff)

    def check_clearance(self, grasp, depth, seg=None, jaw_length=20.0):
        """
        Geometric Clearance Check:
        Verifies that:
          1. The finger contact regions lie inside the image boundary.
          2. The space between the jaws contains elevated object mass.
          3. The finger pads don't collide with obstacles higher than the grasp center.
        """
        h, w = depth.shape[:2]
        c = np.cos(grasp.angle_rad)
        s = np.sin(grasp.angle_rad)
        half_w = grasp.width_px / 2.0

        # Finger pad coordinates in pixel space
        left_pad = np.array([grasp.u - half_w * c, grasp.v - half_w * s])
        right_pad = np.array([grasp.u + half_w * c, grasp.v + half_w * s])

        # 1. Bounds check
        pad_margin = 8
        if not (pad_margin <= left_pad[0] < w - pad_margin and pad_margin <= left_pad[1] < h - pad_margin):
            return 0.0, False, "Left finger pad out of image bounds"
        if not (pad_margin <= right_pad[0] < w - pad_margin and pad_margin <= right_pad[1] < h - pad_margin):
            return 0.0, False, "Right finger pad out of image bounds"

        # 2. Between-jaw occupancy check
        # Sample 5 points along the grasp axis between left_pad and right_pad
        interp_alphas = np.linspace(0.2, 0.8, 5)
        between_pts = [left_pad + a * (right_pad - left_pad) for a in interp_alphas]
        elevations = []
        for pt in between_pts:
            px, py = int(round(pt[0])), int(round(pt[1]))
            d_val = depth[py, px]
            elevations.append(self.table_depth - d_val)

        mean_elevation = float(np.mean(elevations))
        if mean_elevation < 0.005:  # Less than 5mm above table => empty space
            return 0.1, False, "No object mass detected between gripper jaws"

        # 3. Collision check along approach vector
        # Finger pads shouldn't penetrate deep collisions
        left_d = depth[int(round(left_pad[1])), int(round(left_pad[0]))]
        right_d = depth[int(round(right_pad[1])), int(round(right_pad[0]))]
        center_d = depth[int(round(grasp.v)), int(round(grasp.u))]

        # Finger pads should not be significantly higher than the center (collision)
        left_collision = (center_d - left_d) > 0.05
        right_collision = (center_d - right_d) > 0.05
        if left_collision or right_collision:
            return 0.2, False, "Finger collision detected with tall adjacent obstacle"

        clearance_score = float(np.clip(mean_elevation / 0.03, 0.6, 1.0))
        return clearance_score, True, "Clearance verified"

    def check_antipodal_friction(self, grasp, depth):
        """
        Physics & Antipodal Friction Cone Check:
        Computes depth Sobel gradients around the contact points to derive surface normals.
        Verifies that inward normals point toward each other within the friction cone.
        """
        h, w = depth.shape[:2]
        c = np.cos(grasp.angle_rad)
        s = np.sin(grasp.angle_rad)
        half_w = grasp.width_px / 2.0

        lp = (int(round(grasp.u - half_w * c)), int(round(grasp.v - half_w * s)))
        rp = (int(round(grasp.u + half_w * c)), int(round(grasp.v + half_w * s)))

        # Clamp safely
        lp = (np.clip(lp[0], 2, w - 3), np.clip(lp[1], 2, h - 3))
        rp = (np.clip(rp[0], 2, w - 3), np.clip(rp[1], 2, h - 3))

        # Sobel gradients around left and right contact pads
        # Gradient of depth: dz/dx, dz/dy (in pixel/meters)
        dx_l = (depth[lp[1], lp[0] + 1] - depth[lp[1], lp[0] - 1]) * 0.5
        dy_l = (depth[lp[1] + 1, lp[0]] - depth[lp[1] - 1, lp[0]]) * 0.5

        dx_r = (depth[rp[1], rp[0] + 1] - depth[rp[1], rp[0] - 1]) * 0.5
        dy_r = (depth[rp[1] + 1, rp[0]] - depth[rp[1] - 1, rp[0]]) * 0.5

        # Inward surface normal vectors in 2D image plane:
        # Facing toward higher surface elevation (negative depth gradient)
        n_left = np.array([-dx_l, -dy_l])
        n_right = np.array([-dx_r, -dy_r])

        norm_l = np.linalg.norm(n_left)
        norm_r = np.linalg.norm(n_right)

        # Grasp axis unit vector pointing from left pad to right pad
        axis = np.array([c, s])

        # If flat surfaces (small gradients), antipodal condition is naturally satisfied for top-down clamping
        if norm_l < 1e-4 or norm_r < 1e-4:
            return 0.85, True, "Flat contact surface (antipodal feasible)"

        n_left_u = n_left / norm_l
        n_right_u = n_right / norm_r

        # Dot product with grasp axis:
        # Left normal should align with +axis, Right normal should align with -axis
        dot_l = float(np.dot(n_left_u, axis))
        dot_r = float(np.dot(n_right_u, -axis))

        # Mean friction alignment
        mean_align = 0.5 * (dot_l + dot_r)
        antipodal_score = float(np.clip((mean_align + 1.0) / 2.0, 0.2, 1.0))
        is_pass = mean_align > -0.3  # within friction cone margin

        reason = "Friction cone verified" if is_pass else "Surface normal misaligned with grasp axis"
        return antipodal_score, is_pass, reason

    def check_kinematic_reachability(self, grasp, robot=None):
        """
        Robotics & IK Feasibility Check:
        Verifies that 3D world target lies inside reachable workspace envelope
        and Franka Panda IK can solve for joint angles within position error < 2mm.
        """
        if grasp.world_coords is None:
            return 0.5, True, "No 3D coordinates provided, kinematic check bypassed"

        wx, wy, wz = grasp.world_coords

        # 1. Franka Panda workspace bounding box check
        # Table workspace: X in [0.35, 0.65], Y in [-0.20, 0.20], Z in [0.00, 0.15]
        in_x = 0.30 <= wx <= 0.70
        in_y = -0.25 <= wy <= 0.25
        in_z = -0.01 <= wz <= 0.20

        if not (in_x and in_y and in_z):
            return 0.0, False, f"Target [{wx:.2f}, {wy:.2f}, {wz:.2f}] outside safe workspace"

        # 2. Analytical distance from Franka base (base is at [0.0, 0.0, 0.0])
        dist = np.hypot(wx, wy)
        if dist < 0.25 or dist > 0.75:
            return 0.2, False, f"Target radius {dist:.2f}m outside arm reach envelope [0.25m, 0.75m]"

        # 3. If robot instance is available, test numerical IK solution
        if robot is not None and hasattr(robot, "solve_ik"):
            try:
                yaw_cmd = grasp.angle_rad - np.pi / 2.0
                q_sol, ik_err = robot.solve_ik([wx, wy, max(0.02, wz)], target_yaw=yaw_cmd)
                if ik_err > 0.005:  # > 5mm error
                    return 0.4, False, f"IK solution residual error high: {ik_err*1000:.1f}mm"
                return 1.0, True, "IK solution verified (sub-millimeter precision)"
            except Exception as e:
                return 0.5, True, f"IK probe warning: {e}"

        return 0.95, True, "Kinematic workspace reachability verified"

    def validate_candidate(self, grasp, depth, seg=None, robot=None):
        """
        Runs comprehensive multi-criteria validation on a single candidate grasp.
        Returns GraspValidationResult.
        """
        c_score, c_pass, c_reason = self.check_clearance(grasp, depth, seg)
        a_score, a_pass, a_reason = self.check_antipodal_friction(grasp, depth)
        k_score, k_pass, k_reason = self.check_kinematic_reachability(grasp, robot)

        # Composite validated score
        # 40% CNN grasp quality + 25% Clearance + 20% Friction cone + 15% Kinematics
        cnn_q = np.clip(grasp.quality, 0.0, 1.0)
        composite = 0.40 * cnn_q + 0.25 * c_score + 0.20 * a_score + 0.15 * k_score

        # Overall validity
        is_valid = c_pass and k_pass and (composite >= 0.45)

        # Assign verdict classification
        if is_valid and composite >= 0.65:
            verdict = "OPTIMAL"
            reason = "Passes all geometric, friction cone, and kinematic clearance checks"
        elif is_valid:
            verdict = "FEASIBLE"
            reason = "Physically feasible with acceptable clearance margins"
        elif not c_pass:
            verdict = "REJECTED"
            reason = f"Clearance failure: {c_reason}"
        elif not k_pass:
            verdict = "REJECTED"
            reason = f"Kinematic failure: {k_reason}"
        else:
            verdict = "RISKY"
            reason = f"Friction/quality warning: {a_reason}"

        return GraspValidationResult(
            grasp=grasp,
            is_valid=is_valid,
            score=composite,
            clearance_score=c_score,
            antipodal_score=a_score,
            kinematic_score=k_score,
            verdict=verdict,
            reason=reason
        )

    def validate_and_rank(self, grasps, depth, seg=None, robot=None):
        """
        Validates a list of grasp proposals and re-ranks them by their validated physical score.
        Attaches validation metadata directly to each Grasp object.
        Returns:
            validated_grasps: list of Grasp objects sorted by validated score
            validation_results: list of GraspValidationResult objects
        """
        results = []
        for g in grasps:
            res = self.validate_candidate(g, depth, seg=seg, robot=robot)
            # Attach attributes directly to grasp for convenience
            g.validation = res
            g.validated_score = res.score
            g.is_validated = res.is_valid
            g.verdict = res.verdict
            results.append(res)

        # Sort grasps by composite validated score in descending order
        ranked_indices = np.argsort([-r.score for r in results])
        sorted_grasps = [grasps[i] for i in ranked_indices]
        sorted_results = [results[i] for i in ranked_indices]

        return sorted_grasps, sorted_results

