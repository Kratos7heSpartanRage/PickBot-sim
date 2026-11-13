"""
=============================================================================
  AI-Based Vision-Guided Robotic Grasping System: Deliverable 2
  Environment & Scene Synthesis Subsystem: Grasping Environment
  Lead: Satyam Kumar
=============================================================================
PyBullet Simulated Picking Environment:
  - Studio lighting with soft shadow mapping (lightPosition=[1.5, -1.2, 2.5],
    shadowMapResolution=4096, shadowMapIntensity=0.35).
  - High-precision Z-calibration: Table top surface calibrated to exactly Z=0.000m.
  - Workspace boundary mat: Subtle dark mat (0.5mm above table) with crisp
    colored border lines over X in [0.35, 0.65], Y in [-0.20, 0.20].
  - Collection drop tray at [0.45, 0.45, 0.0] with fixed base, out of overhead
    camera FOV.
  - Strict observation contract (RGB, metric depth, segmentation mask).
=============================================================================
"""
import time
import numpy as np
import pybullet as p
import pybullet_data

from .camera import OverheadCamera
from .object_spawner import ObjectSpawner
from .robot import FrankaPandaRobot


class GraspingEnvironment:
    """
    Calibrated PyBullet picking environment fulfilling Deliverable 2 specifications.
    """
    def __init__(self, gui=False):
        self.gui = gui
        connection_mode = p.GUI if self.gui else p.DIRECT
        self.client_id = p.connect(connection_mode)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / 240.0)

        # 1. Studio Lighting, Soft Shadows, and 3D Visualizer Configuration
        if self.gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=1.4,
                cameraYaw=50.0,
                cameraPitch=-35.0,
                cameraTargetPosition=[0.48, 0.05, 0.10]
            )
            # Eval 2 spec: studio lighting & soft shadow parameters
            p.configureDebugVisualizer(
                p.COV_ENABLE_SHADOWS, 1,
                lightPosition=[1.5, -1.2, 2.5],
                shadowMapResolution=4096,
                shadowMapIntensity=0.35
            )
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)

        # 2. Environment Assets & Rock-Solid Z-Calibration
        # Calibrated table basePosition [0.5, 0.0, -0.626] puts table top surface at Z=0.000m
        self.table_top_z = 0.0
        self.plane_id = p.loadURDF("plane.urdf", [0.0, 0.0, -0.626])
        self.table_id = p.loadURDF("table/table.urdf", [0.5, 0.0, -0.626], useFixedBase=True)

        # High-friction table contact physics
        p.changeDynamics(
            self.table_id, -1,
            lateralFriction=1.20,
            spinningFriction=0.02,
            rollingFriction=0.01,
            restitution=0.02
        )
        # Warm industrial slate table top visual color
        p.changeVisualShape(self.table_id, -1, rgbaColor=[0.70, 0.70, 0.72, 1.0])

        # 3. Workspace Boundary Mat (Height 0.5mm above table: Z=0.00025)
        # Active pick area: X in [0.35, 0.65], Y in [-0.20, 0.20]
        mat_half_x = (0.65 - 0.35) / 2.0  # 0.15m
        mat_half_y = (0.20 - (-0.20)) / 2.0  # 0.20m
        mat_half_z = 0.00025  # 0.5mm total thickness
        vis_mat = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[mat_half_x, mat_half_y, mat_half_z],
            rgbaColor=[0.18, 0.20, 0.24, 1.0]  # Subtle dark charcoal mat
        )
        self.mat_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=vis_mat,
            basePosition=[0.50, 0.00, mat_half_z]
        )

        # 4. Collection Drop Tray at [0.45, 0.45, table_z]
        # Scaled to 0.4 so it remains outside the overhead camera FOV while within robot reach
        try:
            self.tray_id = p.loadURDF(
                "tray/traybox.urdf",
                [0.45, 0.45, self.table_top_z],
                globalScaling=0.40,
                useFixedBase=True
            )
            p.changeVisualShape(self.tray_id, -1, rgbaColor=[0.28, 0.30, 0.34, 1.0])
        except Exception:
            self.tray_id = None

        # 5. Crisp Workspace Accent Lines in GUI
        if self.gui:
            self._draw_workspace_bounds()

        # 6. Initialize Robot Arm (Home tucked away outside camera FOV)
        self.robot = FrankaPandaRobot(base_pos=(0.0, 0.0, 0.0))

        # 7. Initialize Overhead Camera (z=0.85m directly above center [0.5, 0.0])
        self.camera = OverheadCamera(
            target_pos=(0.5, 0.0, 0.0),
            camera_pos=(0.5, 0.0, 0.85),
            img_size=(300, 300)
        )

        # 8. Initialize Calibrated Object Spawner
        self.spawner = ObjectSpawner(
            workspace_bounds=((0.37, 0.63), (-0.17, 0.17)),
            table_height=self.table_top_z
        )

    def _draw_workspace_bounds(self):
        """Draws crisp accent border lines around active picking workspace."""
        x_lo, x_hi = 0.35, 0.65
        y_lo, y_hi = -0.20, 0.20
        z = 0.002
        accent_color = [0.08, 0.88, 0.92]  # Crisp Electric Cyan
        line_w = 2.5
        p.addUserDebugLine([x_lo, y_lo, z], [x_hi, y_lo, z], lineColorRGB=accent_color, lineWidth=line_w, lifeTime=0)
        p.addUserDebugLine([x_hi, y_lo, z], [x_hi, y_hi, z], lineColorRGB=accent_color, lineWidth=line_w, lifeTime=0)
        p.addUserDebugLine([x_hi, y_hi, z], [x_lo, y_hi, z], lineColorRGB=accent_color, lineWidth=line_w, lifeTime=0)
        p.addUserDebugLine([x_lo, y_hi, z], [x_lo, y_lo, z], lineColorRGB=accent_color, lineWidth=line_w, lifeTime=0)

    def reset(self, num_objects=3):
        """Resets robot arm to neutral tucked home pose and spawns calibrated objects."""
        self.robot.reset()
        spawned_ids = self.spawner.spawn_random_objects(num_objects=num_objects)
        return spawned_ids

    def get_observation(self):
        """
        Captures overhead RGB, metric Depth, and strictly compliant Segmentation mask.
        Contract (Section 4.5):
          rgb   : np.ndarray (uint8)   [300, 300, 3] in [0, 255]
          depth : np.ndarray (float32) [300, 300] linear meters (table at 0.85m +- 0.01m)
          seg   : np.ndarray (int32)   [300, 300] (1=table, 0=floor, body_id=object)
        """
        rgb, depth, seg_raw = self.camera.capture()

        # Remap segmentation mask to maintain standard contract:
        # Table surface and workspace mat -> 1
        # Floor plane -> 0
        # Specific objects -> retain exact object body_id
        seg = seg_raw.copy()
        if hasattr(self, 'mat_id') and self.mat_id is not None:
            seg[seg_raw == self.mat_id] = 1
        if hasattr(self, 'tray_id') and self.tray_id is not None:
            seg[seg_raw == self.tray_id] = 1
        if hasattr(self, 'plane_id') and self.plane_id is not None:
            seg[seg_raw == self.plane_id] = 0
        if hasattr(self, 'table_id') and self.table_id is not None:
            seg[seg_raw == self.table_id] = 1

        return rgb, depth, seg

    def get_ground_truth_grasps(self):
        """Returns ground-truth Grasp objects for all currently spawned objects."""
        return self.spawner.get_ground_truth_grasps(self.camera)

    def step(self):
        p.stepSimulation()

    def close(self):
        try:
            p.disconnect(self.client_id)
        except Exception:
            pass
