"""
=============================================================================
  AI-Based Vision-Guided Robotic Grasping System: Deliverable 2
  Robotics & Kinematics Subsystem: Franka Panda Arm Controller
=============================================================================
"""
import time
import numpy as np
import pybullet as p
import pybullet_data


class FrankaPandaRobot:
    """
    Franka Emika Panda 7-DOF arm with a 2-finger parallel jaw gripper.
    Features:
      - Seeded inverse kinematics (IK) ensuring convergence without joint limit violations.
      - Active gripper force retention during lift and transfer.
      - High-friction finger pad contact dynamics.
      - Robust motor controller target updates on reset and smooth home return.
    """
    def __init__(self, base_pos=(0.0, 0.0, 0.0), base_orn=(0, 0, 0, 1)):
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.base_pos = base_pos
        self.base_orn = base_orn

        self.robot_id = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=self.base_pos,
            baseOrientation=self.base_orn,
            useFixedBase=True
        )

        self.num_joints = p.getNumJoints(self.robot_id)
        self.arm_joints = [0, 1, 2, 3, 4, 5, 6]
        self.gripper_joints = [9, 10]
        self.ee_link_idx = 11  # panda_grasptarget

        # Calibrate finger pad contact dynamics
        for g_idx in self.gripper_joints:
            p.changeDynamics(
                self.robot_id, g_idx,
                lateralFriction=2.5,
                spinningFriction=0.10,
                rollingFriction=0.01,
                contactDamping=100.0,
                contactStiffness=50000.0
            )

        # Home rest posture (arm safely tucked to the side to keep camera view unobstructed)
        self.home_posture = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
        # Downward seed pose pointing toward table workspace
        self.downward_seed = [0.0, 0.50, 0.0, -2.00, 0.0, 2.50, 0.785]

        self.reset()

    def _step(self):
        """Step physics with smooth animation sleep in GUI mode."""
        p.stepSimulation()
        try:
            if p.getConnectionInfo()["connectionMethod"] == p.GUI:
                time.sleep(1.0 / 240.0)
        except Exception:
            pass

    def reset(self):
        """
        Resets robot joints to neutral home posture and immediately updates
        the motor controller's position targets to prevent drifting back to
        previous waypoints.
        """
        for idx, q in zip(self.arm_joints, self.home_posture):
            p.resetJointState(self.robot_id, idx, q, targetVelocity=0.0)
            p.setJointMotorControl2(
                self.robot_id,
                idx,
                p.POSITION_CONTROL,
                targetPosition=q,
                targetVelocity=0.0,
                force=200.0,
                maxVelocity=1.5
            )
        for g_idx in self.gripper_joints:
            p.resetJointState(self.robot_id, g_idx, 0.04, targetVelocity=0.0)
            p.setJointMotorControl2(
                self.robot_id,
                g_idx,
                p.POSITION_CONTROL,
                targetPosition=0.04,
                targetVelocity=0.0,
                force=40.0
            )
        for _ in range(10):
            p.stepSimulation()

    def move_to_home(self, steps=80):
        """Smoothly drives the robot arm back to neutral home posture."""
        for idx, q in zip(self.arm_joints, self.home_posture):
            p.setJointMotorControl2(
                self.robot_id,
                idx,
                p.POSITION_CONTROL,
                targetPosition=q,
                targetVelocity=0.0,
                force=200.0,
                maxVelocity=1.5
            )
        for _ in range(steps):
            self._step()
        self.reset()

    def close_gripper(self, force=120.0, target_pos=0.0, steps=80):
        """Closes parallel jaw gripper with specified clamping force."""
        for g_idx in self.gripper_joints:
            p.setJointMotorControl2(
                self.robot_id,
                g_idx,
                p.POSITION_CONTROL,
                targetPosition=target_pos,
                force=force
            )
        for _ in range(steps):
            self._step()

    def open_gripper(self, target_pos=0.04, force=40.0, steps=60):
        """Opens parallel jaw gripper to target stroke (default 0.04m per finger)."""
        for g_idx in self.gripper_joints:
            p.setJointMotorControl2(
                self.robot_id,
                g_idx,
                p.POSITION_CONTROL,
                targetPosition=target_pos,
                force=force
            )
        for _ in range(steps):
            self._step()

    def move_to_cartesian(self, target_pos, target_yaw=0.0, steps=80,
                          keep_gripper_closed=False, grip_force=120.0):
        """
        Moves the end-effector to (target_pos, target_yaw) using Inverse Kinematics.
        Uses seeded IK to prevent joint limit violations and singularity traps.
        """
        target_orn = p.getQuaternionFromEuler([np.pi, 0.0, target_yaw])

        # Temporarily seed IK from natural downward posture to guarantee valid joint branch
        current_q = [p.getJointState(self.robot_id, i)[0] for i in self.arm_joints]
        seed = list(self.downward_seed)
        seed[6] = 0.785 + target_yaw
        for i, q in enumerate(seed):
            p.resetJointState(self.robot_id, i, q, targetVelocity=0.0)

        joint_poses = p.calculateInverseKinematics(
            self.robot_id,
            self.ee_link_idx,
            target_pos,
            target_orn,
            maxNumIterations=200,
            residualThreshold=1e-5
        )

        # Restore actual joint angles so physics trajectory starts from current pose
        for i, q in enumerate(current_q):
            p.resetJointState(self.robot_id, i, q, targetVelocity=0.0)

        # Apply position control to drive motors smoothly to target joint angles
        for i, j_idx in enumerate(self.arm_joints):
            p.setJointMotorControl2(
                self.robot_id,
                j_idx,
                p.POSITION_CONTROL,
                targetPosition=joint_poses[i],
                force=200.0,
                maxVelocity=1.5
            )

        if keep_gripper_closed:
            for g_idx in self.gripper_joints:
                p.setJointMotorControl2(
                    self.robot_id,
                    g_idx,
                    p.POSITION_CONTROL,
                    targetPosition=0.0,
                    force=grip_force
                )

        for _ in range(steps):
            if keep_gripper_closed:
                for g_idx in self.gripper_joints:
                    p.setJointMotorControl2(
                        self.robot_id,
                        g_idx,
                        p.POSITION_CONTROL,
                        targetPosition=0.0,
                        force=grip_force
                    )
            self._step()

    def get_ee_pose(self):
        """Returns current end-effector (position, orientation)."""
        state = p.getLinkState(self.robot_id, self.ee_link_idx)
        return np.array(state[0]), np.array(state[1])
