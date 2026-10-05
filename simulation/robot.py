import time

import numpy as np
import pybullet as p
import pybullet_data


class FrankaPandaRobot:
    """
    Franka Emika Panda 7-DOF arm with a 2-finger parallel jaw gripper.

    Motion model (smooth by construction):
      * Cartesian moves are split into straight-line waypoints (~1 cm apart).
      * Each waypoint is solved with an *iterative, seeded* IK (sub-mm accuracy).
      * The joint trajectory is executed with a minimum-jerk time profile,
        streaming small per-step POSITION_CONTROL targets (no teleports).
      * In GUI mode each physics step is paced to wall-clock time so the
        motion is rendered at real speed instead of flashing past.
    """

    # Panda link indices (pybullet_data/franka_panda/panda.urdf)
    ARM_JOINTS = [0, 1, 2, 3, 4, 5, 6]
    FINGER_JOINTS = [9, 10]
    EE_LINK = 11                 # panda_grasptarget: ~7 mm above fingertip pads
    ARM_FORCES = [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0]   # Nm (real Panda limits)
    FINGER_OPEN = 0.04           # m per finger (0.08 m total opening)

    def __init__(self, base_pos=(0.0, 0.0, 0.0), base_orn=(0, 0, 0, 1), realtime=False, step_hook=None):
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.base_pos = base_pos
        self.base_orn = base_orn
        self.realtime = realtime          # pace sim to wall-clock (GUI only)
        self.step_hook = step_hook        # optional callable run after every sim step
        self.dt = 1.0 / 240.0
        self.verbose = False              # print per-move IK diagnostics

        self.robot_id = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=self.base_pos,
            baseOrientation=self.base_orn,
            useFixedBase=True,
        )

        self.num_joints = p.getNumJoints(self.robot_id)
        self.arm_joints = list(self.ARM_JOINTS)
        self.gripper_joints = list(self.FINGER_JOINTS)
        self.ee_link_idx = self.EE_LINK

        # Movable DOF (arm + fingers) used by calculateInverseKinematics
        self.movable = [j for j in range(self.num_joints)
                        if p.getJointInfo(self.robot_id, j)[2] != p.JOINT_FIXED]
        self.lower = [p.getJointInfo(self.robot_id, j)[8] for j in self.movable]
        self.upper = [p.getJointInfo(self.robot_id, j)[9] for j in self.movable]
        self.ranges = [u - l for l, u in zip(self.lower, self.upper)]

        # Home posture: arm folded back toward its base. Verified to be fully
        # OUTSIDE the overhead camera frustum (0 robot pixels in the image).
        self.home_posture = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
        # Null-space rest pose biases IK toward an elbow-up, wrist-down config
        self.rest_pose = [0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.785, self.FINGER_OPEN, self.FINGER_OPEN]

        # Grippy fingertips (default Panda fingers are slippery in PyBullet)
        for f in self.gripper_joints:
            p.changeDynamics(self.robot_id, f, lateralFriction=2.0, spinningFriction=0.02,
                             rollingFriction=0.001, frictionAnchor=True)

        self.gripper_target = self.FINGER_OPEN
        self.grip_force = 40.0
        self.reset()

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------
    def _step(self):
        p.stepSimulation()
        if self.step_hook is not None:
            self.step_hook()
        if self.realtime:
            time.sleep(self.dt)

    def step(self, n=1):
        for _ in range(n):
            self._step()

    def get_arm_q(self):
        return np.array([p.getJointState(self.robot_id, j)[0] for j in self.arm_joints])

    def _hold_gripper(self):
        p.setJointMotorControlArray(
            self.robot_id, self.gripper_joints, p.POSITION_CONTROL,
            targetPositions=[self.gripper_target] * 2, forces=[self.grip_force] * 2)

    def _command_arm(self, q):
        kwargs = dict(targetPositions=list(q), forces=self.ARM_FORCES,
                      positionGains=[0.12] * 7, velocityGains=[1.0] * 7)
        p.setJointMotorControlArray(self.robot_id, self.arm_joints, p.POSITION_CONTROL, **kwargs)

    def reset(self):
        """Instantly resets robot joints to home (used only at scene setup)."""
        for idx, q in zip(self.arm_joints, self.home_posture):
            p.resetJointState(self.robot_id, idx, q)
        for g_idx in self.gripper_joints:
            p.resetJointState(self.robot_id, g_idx, self.FINGER_OPEN)
        self.gripper_target = self.FINGER_OPEN
        self._command_arm(self.home_posture)
        self._hold_gripper()

    # ------------------------------------------------------------------
    # Inverse kinematics
    # ------------------------------------------------------------------
    @staticmethod
    def down_orientation(yaw):
        """Top-down gripper orientation. NOTE: the jaw closing axis lies at world angle yaw + 90 deg."""
        return p.getQuaternionFromEuler([np.pi, 0.0, yaw])

    def solve_ik(self, target_pos, target_yaw=0.0, seed=None, max_iters=20, tol=5e-4):
        """
        Iterative, seeded IK. PyBullet's single-shot IK can leave >1 cm error,
        so we repeatedly re-solve from the previous solution (evaluated with
        forward kinematics) until the end-effector is within `tol` metres.
        The real joint state is saved and restored, so this is side-effect free.
        Returns (arm_q[7], position_error_m).
        """
        target_pos = np.asarray(target_pos, dtype=float)
        target_orn = self.down_orientation(target_yaw)

        saved = [p.getJointState(self.robot_id, j)[:2] for j in self.movable]
        if seed is not None:
            for j, q in zip(self.arm_joints, seed):
                p.resetJointState(self.robot_id, j, q)

        q_sol, err = None, np.inf
        for _ in range(max_iters):
            q_all = p.calculateInverseKinematics(
                self.robot_id, self.ee_link_idx, target_pos.tolist(), target_orn,
                lowerLimits=self.lower, upperLimits=self.upper, jointRanges=self.ranges,
                restPoses=self.rest_pose, maxNumIterations=200, residualThreshold=1e-5)
            q_sol = np.array(q_all[:7])
            for j, q in zip(self.arm_joints, q_sol):
                p.resetJointState(self.robot_id, j, q)
            ee = np.array(p.getLinkState(self.robot_id, self.ee_link_idx,
                                         computeForwardKinematics=True)[4])
            err = float(np.linalg.norm(ee - target_pos))
            if err < tol:
                break

        for j, (q, qd) in zip(self.movable, saved):
            p.resetJointState(self.robot_id, j, q, qd)
        return q_sol, err

    # ------------------------------------------------------------------
    # Smooth motion
    # ------------------------------------------------------------------
    @staticmethod
    def _min_jerk(s):
        return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5

    def _execute_joint_path(self, path, steps, settle_steps=60, settle_tol=1e-3):
        """
        Streams a joint-space path (list of 7-vectors) to the motors with a
        minimum-jerk time profile over `steps` physics steps, then lets the
        controller settle.
        """
        path = np.asarray(path)
        n_seg = len(path) - 1
        for k in range(1, steps + 1):
            s = self._min_jerk(k / steps) * n_seg
            i = min(int(s), n_seg - 1)
            q = path[i] + (path[i + 1] - path[i]) * (s - i)
            self._command_arm(q)
            self._hold_gripper()
            self._step()
        for _ in range(settle_steps):
            if np.max(np.abs(self.get_arm_q() - path[-1])) < settle_tol:
                break
            self._step()

    def move_to_cartesian(self, target_pos, target_yaw=0.0, steps=None, speed=0.20, label=None):
        """
        Smooth straight-line Cartesian move of the grasp target frame.
        `steps` is a lower bound on motion duration (physics steps @240 Hz);
        the duration is also scaled by distance / `speed` (m/s).
        Returns a dict with commanded vs. achieved pose for diagnostics.
        """
        target_pos = np.asarray(target_pos, dtype=float)
        start_pos, _ = self.get_ee_pose()
        dist = float(np.linalg.norm(target_pos - start_pos))

        n_wp = max(2, int(np.ceil(dist / 0.01)) + 1)
        start_yaw = self._current_yaw()
        dyaw = (target_yaw - start_yaw + np.pi) % (2 * np.pi) - np.pi
        q_prev = self.get_arm_q()
        path = [q_prev]
        ik_err = 0.0
        for k in range(1, n_wp):
            a = k / (n_wp - 1)
            wp = start_pos + (target_pos - start_pos) * a
            q_prev, ik_err = self.solve_ik(wp, start_yaw + dyaw * a, seed=q_prev)
            path.append(q_prev)

        dur = max(int(steps or 0), int(dist / speed / self.dt), int(abs(dyaw) / 1.5 / self.dt), 30)
        self._execute_joint_path(path, dur)

        actual, _ = self.get_ee_pose()
        result = {
            "commanded": target_pos,
            "actual": actual,
            "ik_error_m": ik_err,
            "tracking_error_m": float(np.linalg.norm(actual - target_pos)),
        }
        if self.verbose:
            tag = f"{label} " if label else ""
            print(f"          │ {tag}IK  cmd=[{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]"
                  f"  act=[{actual[0]:.3f}, {actual[1]:.3f}, {actual[2]:.3f}]"
                  f"  ik_err={ik_err*1000:.1f}mm  track_err={result['tracking_error_m']*1000:.1f}mm")
        return result

    def move_to_joints(self, q_target, steps=None, max_joint_speed=1.0):
        """Smooth joint-space move (used for going home)."""
        q0 = self.get_arm_q()
        q_target = np.asarray(q_target, dtype=float)
        dur = max(int(steps or 0), int(np.max(np.abs(q_target - q0)) / max_joint_speed / self.dt), 30)
        self._execute_joint_path([q0, q_target], dur)

    def go_home(self, steps=None):
        self.move_to_joints(self.home_posture, steps=steps)

    def jog_to(self, target_pos, target_yaw=0.0):
        """Non-blocking teleop: set motor targets toward a pose (caller keeps stepping)."""
        q, _ = self.solve_ik(target_pos, target_yaw, seed=self.get_arm_q(), max_iters=3)
        p.setJointMotorControlArray(self.robot_id, self.arm_joints, p.POSITION_CONTROL,
                                    targetPositions=list(q), forces=self.ARM_FORCES,
                                    positionGains=[0.03] * 7)
        self._hold_gripper()

    def _current_yaw(self):
        _, orn = self.get_ee_pose()
        rot = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        # With orientation euler(pi, 0, yaw), local X axis = (cos yaw, sin yaw, 0)
        return float(np.arctan2(rot[1, 0], rot[0, 0]))

    # ------------------------------------------------------------------
    # Gripper
    # ------------------------------------------------------------------
    def open_gripper(self, steps=60):
        self.gripper_target = self.FINGER_OPEN
        self.grip_force = 40.0
        self._hold_gripper()
        self.step(steps)

    def close_gripper(self, force=60.0, steps=120, object_ids=None, monitor=None):
        """
        Closes the fingers with a force-limited position controller and
        returns the peak normal contact force measured on the fingers.
        `monitor(force)` is called every 5 steps (for live GUI read-outs).
        """
        self.gripper_target = 0.0
        self.grip_force = force   # keeps squeezing with this force during later arm motion
        self._hold_gripper()
        peak = 0.0
        for k in range(steps):
            self._step()
            if k % 5 == 0:
                f = self.get_finger_contact_force(object_ids)
                peak = max(peak, f)
                if monitor is not None:
                    monitor(f)
        return peak

    def get_finger_contact_force(self, object_ids=None):
        """Sum of normal contact forces between the finger links and (optionally) given bodies."""
        total = 0.0
        for link in self.gripper_joints:
            for c in p.getContactPoints(bodyA=self.robot_id, linkIndexA=link):
                if object_ids is None or c[2] in object_ids:
                    total += abs(c[9])
        return total

    def gripper_opening(self):
        return sum(p.getJointState(self.robot_id, j)[0] for j in self.gripper_joints)

    def get_ee_pose(self):
        """Returns current grasp-target frame (position, orientation)."""
        state = p.getLinkState(self.robot_id, self.ee_link_idx, computeForwardKinematics=True)
        return np.array(state[4]), np.array(state[5])
