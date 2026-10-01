import numpy as np
import pybullet as p
import pybullet_data
import os

class FrankaPandaRobot:
    """
    Franka Emika Panda 7-DOF arm with a 2-finger parallel jaw gripper.
    Supports inverse kinematics (IK), Cartesian waypoint positioning,
    and gripper opening/closing for Deliverable 1 demonstration and Deliverable 2 execution.
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

        # Panda arm joints: 0 to 6 are arm revolute joints
        # 7, 8 are wrist/hand
        # 9, 10 are parallel gripper finger joints (prismatic)
        self.num_joints = p.getNumJoints(self.robot_id)
        self.arm_joints = [0, 1, 2, 3, 4, 5, 6]
        self.gripper_joints = [9, 10]
        self.ee_link_idx = 11  # panda_grasptarget or hand

        # Home rest posture (arm safely tucked to the side to keep camera view unobstructed)
        self.home_posture = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
        self.reset()

    def reset(self):
        """Resets robot joints to neutral home posture."""
        for idx, q in zip(self.arm_joints, self.home_posture):
            p.resetJointState(self.robot_id, idx, q)
        # Open gripper fingers (0.04m each = 0.08m total opening)
        for g_idx in self.gripper_joints:
            p.resetJointState(self.robot_id, g_idx, 0.04)

    def move_to_cartesian(self, target_pos, target_yaw=0.0, steps=80):
        """
        Uses Inverse Kinematics to move the end-effector to (target_pos, target_yaw).
        Grip orientation faces top-down (pointing downward toward the table).
        """
        # Gripper pointing down [roll=pi, pitch=0, yaw=target_yaw]
        target_orn = p.getQuaternionFromEuler([np.pi, 0.0, target_yaw])

        joint_poses = p.calculateInverseKinematics(
            self.robot_id,
            self.ee_link_idx,
            target_pos,
            target_orn,
            maxNumIterations=100,
            residualThreshold=1e-4
        )

        # Apply position control on arm joints
        for i, j_idx in enumerate(self.arm_joints):
            p.setJointMotorControl2(
                self.robot_id,
                j_idx,
                p.POSITION_CONTROL,
                targetPosition=joint_poses[i],
                force=150.0,
                maxVelocity=1.5
            )

        for _ in range(steps):
            p.stepSimulation()

    def get_ee_pose(self):
        """Returns current end-effector (position, orientation)."""
        state = p.getLinkState(self.robot_id, self.ee_link_idx)
        return np.array(state[0]), np.array(state[1])
