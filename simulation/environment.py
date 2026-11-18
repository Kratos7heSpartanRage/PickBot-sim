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
    def __init__(self, gui=False, enable_gui_panels=False):
        self.gui = gui
        self.enable_gui_panels = enable_gui_panels
        connection_mode = p.GUI if self.gui else p.DIRECT
        self.client_id = p.connect(connection_mode)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / 240.0)

        # 1. Studio Lighting, Soft Shadows, and 3D Visualizer Configuration
        if self.gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=1.05,
                cameraYaw=48.0,
                cameraPitch=-28.0,
                cameraTargetPosition=[0.48, 0.08, 0.12]
            )
            # Eval 2 spec: studio lighting & soft shadow parameters
            p.configureDebugVisualizer(
                p.COV_ENABLE_SHADOWS, 1,
                lightPosition=[1.5, -1.2, 2.5],
                shadowMapResolution=4096,
                shadowMapIntensity=0.35
            )
            # Clean Studio Mode: Disable buggy 2D C++ ExampleBrowser and float sliders by default
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1 if self.enable_gui_panels else 0)
            p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)

        # 2. Environment Assets & Rock-Solid Z-Calibration
        # Calibrated table basePosition [0.5, 0.0, -0.626] puts table top surface at Z=0.000m
        self.table_top_z = 0.0
        self.plane_id = p.loadURDF("plane.urdf", [0.0, 0.0, -0.626])
        p.changeVisualShape(self.plane_id, -1, rgbaColor=[0.14, 0.16, 0.18, 1.0])

        # Industrial Workcell Base Platform under robot and table
        vis_platform = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.85, 0.70, 0.005],
            rgbaColor=[0.20, 0.22, 0.25, 1.0]  # Dark matte epoxy platform
        )
        self.platform_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=vis_platform,
            basePosition=[0.35, 0.12, -0.626 + 0.005]
        )

        # Industrial Robot Mounting Pedestal (Extends from floor Z=-0.626 to robot base Z=0.000)
        pedestal_h = 0.626
        vis_pedestal = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.13,
            length=pedestal_h,
            rgbaColor=[0.22, 0.24, 0.26, 1.0]  # Dark industrial anodized aluminum
        )
        col_pedestal = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=0.13,
            height=pedestal_h
        )
        self.pedestal_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col_pedestal,
            baseVisualShapeIndex=vis_pedestal,
            basePosition=[0.0, 0.0, -pedestal_h / 2.0]
        )

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

        # 5. Authentic Automated Manufacturing Workcell / Factory Environment Setting
        self._setup_factory_environment()

        # 6. Crisp Workspace Accent Lines in GUI
        if self.gui:
            self._draw_workspace_bounds()

        # 7. Initialize Robot Arm (Home tucked away outside camera FOV)
        self.robot = FrankaPandaRobot(base_pos=(0.0, 0.0, 0.0))

        # 8. Initialize Overhead Camera (z=0.85m directly above center [0.5, 0.0])
        self.camera = OverheadCamera(
            target_pos=(0.5, 0.0, 0.0),
            camera_pos=(0.5, 0.0, 0.85),
            img_size=(300, 300)
        )

        # 9. Initialize Calibrated Object Spawner
        self.spawner = ObjectSpawner(
            workspace_bounds=((0.37, 0.63), (-0.17, 0.17)),
            table_height=self.table_top_z
        )

    def _setup_factory_environment(self):
        """
        Constructs an authentic automated manufacturing workcell / factory environment:
          1. Architectural Factory Walls (Light industrial RAL 7035 panels with slate baseboards & safety stripe).
          2. Safety Enclosure (Aluminum extrusion posts + translucent polycarbonate protective shielding).
          3. Overhead Sensor Gantry (Rigid aluminum arch holding the downward-facing RGB-D camera).
          4. Industrial Robot Controller Cabinet (KUKA/Franka PLC electrical unit with E-Stop).
          5. 3-Tier Andon Status Light Tower (Green / Amber / Red indicators).
          6. Collection Tray Outfeed Ramp (Steel industrial chute leading to facility staging).
        All fixtures use mass=0 and collision=-1 so they add zero physics overhead and cannot impede robot motions.
        """
        fl_z = -0.626  # Ground plane elevation

        # 1. Architectural Factory Walls
        # Rear wall
        vis_rear = p.createVisualShape(p.GEOM_BOX, halfExtents=[2.4, 0.04, 1.4], rgbaColor=[0.82, 0.84, 0.86, 1.0])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vis_rear, basePosition=[0.45, -1.35, 0.77 + fl_z])
        # Baseboard plinth (dark kickplate)
        vis_base = p.createVisualShape(p.GEOM_BOX, halfExtents=[2.4, 0.05, 0.08], rgbaColor=[0.20, 0.22, 0.25, 1.0])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vis_base, basePosition=[0.45, -1.34, 0.08 + fl_z])
        # Industrial Safety Blue accent stripe on wall
        vis_stripe = p.createVisualShape(p.GEOM_BOX, halfExtents=[2.4, 0.045, 0.045], rgbaColor=[0.12, 0.38, 0.58, 1.0])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vis_stripe, basePosition=[0.45, -1.34, 0.65])

        # Left partition wall
        vis_left = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.04, 1.8, 1.4], rgbaColor=[0.76, 0.78, 0.80, 1.0])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vis_left, basePosition=[-0.95, 0.20, 0.77 + fl_z])
        vis_left_base = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05, 1.8, 0.08], rgbaColor=[0.20, 0.22, 0.25, 1.0])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vis_left_base, basePosition=[-0.94, 0.20, 0.08 + fl_z])

        # 2. Safety Fence Enclosure (Aluminum Framing + Translucent Polycarbonate)
        post_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.025, 0.025, 0.65], rgbaColor=[0.30, 0.33, 0.36, 1.0])
        posts = [
            [1.20, -0.65], [1.20, 0.85],
            [-0.45, 0.85], [-0.45, -0.65]
        ]
        for px_c, py_c in posts:
            p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=post_shape, basePosition=[px_c, py_c, 0.65 + fl_z])

        # Translucent protective glass shields (Rear and Right side)
        glass_rear = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.80, 0.01, 0.45], rgbaColor=[0.28, 0.42, 0.56, 0.32])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=glass_rear, basePosition=[0.38, -0.65, 0.65 + fl_z])
        glass_right = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.01, 0.75, 0.45], rgbaColor=[0.28, 0.42, 0.56, 0.32])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=glass_right, basePosition=[1.20, 0.10, 0.65 + fl_z])

        # Safety kickplates (Yellow hazard baseboards along fence)
        vis_kick_r = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.80, 0.012, 0.05], rgbaColor=[0.92, 0.78, 0.12, 1.0])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vis_kick_r, basePosition=[0.38, -0.65, 0.05 + fl_z])
        vis_kick_side = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.012, 0.75, 0.05], rgbaColor=[0.92, 0.78, 0.12, 1.0])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vis_kick_side, basePosition=[1.20, 0.10, 0.05 + fl_z])

        # 3. Overhead Inspection Camera Gantry (Above picking table)
        vis_mast = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.52], rgbaColor=[0.26, 0.28, 0.30, 1.0])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vis_mast, basePosition=[0.50, -0.42, 0.52])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vis_mast, basePosition=[0.50, 0.58, 0.52])
        # Crossbeam
        vis_beam = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.022, 0.50, 0.022], rgbaColor=[0.24, 0.26, 0.28, 1.0])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vis_beam, basePosition=[0.50, 0.08, 1.04])
        # Physical Camera Sensor Enclosure (RealSense / Zivid style)
        vis_cambox = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.045, 0.045, 0.035], rgbaColor=[0.10, 0.12, 0.14, 1.0])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vis_cambox, basePosition=[0.50, 0.0, 0.98])
        vis_camlens = p.createVisualShape(p.GEOM_CYLINDER, radius=0.018, length=0.015, rgbaColor=[0.05, 0.05, 0.08, 1.0])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vis_camlens, basePosition=[0.50, 0.0, 0.94])

        # 4. Industrial Control Cabinet (KUKA/Franka PLC electrical enclosure)
        vis_cab = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.22, 0.26, 0.45], rgbaColor=[0.72, 0.74, 0.76, 1.0])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vis_cab, basePosition=[-0.55, -0.85, 0.45 + fl_z])
        vis_cab_top = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.23, 0.27, 0.02], rgbaColor=[0.18, 0.20, 0.22, 1.0])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vis_cab_top, basePosition=[-0.55, -0.85, 0.92 + fl_z])
        # E-Stop on Cabinet
        vis_estop_base = p.createVisualShape(p.GEOM_CYLINDER, radius=0.035, length=0.025, rgbaColor=[0.95, 0.80, 0.10, 1.0])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vis_estop_base, basePosition=[-0.65, -0.80, 0.95 + fl_z])
        vis_estop_btn = p.createVisualShape(p.GEOM_CYLINDER, radius=0.024, length=0.020, rgbaColor=[0.92, 0.15, 0.15, 1.0])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vis_estop_btn, basePosition=[-0.65, -0.80, 0.97 + fl_z])

        # 5. 3-Tier Andon Light Stack
        vis_mast_lt = p.createVisualShape(p.GEOM_CYLINDER, radius=0.012, length=0.35, rgbaColor=[0.22, 0.24, 0.26, 1.0])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vis_mast_lt, basePosition=[-0.45, -0.95, 1.10 + fl_z])
        # Green (Running / Online)
        vis_lt_g = p.createVisualShape(p.GEOM_CYLINDER, radius=0.022, length=0.05, rgbaColor=[0.10, 0.95, 0.25, 0.95])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vis_lt_g, basePosition=[-0.45, -0.95, 1.30 + fl_z])
        # Amber (Standby)
        vis_lt_a = p.createVisualShape(p.GEOM_CYLINDER, radius=0.022, length=0.05, rgbaColor=[0.95, 0.65, 0.10, 0.95])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vis_lt_a, basePosition=[-0.45, -0.95, 1.36 + fl_z])
        # Red (Alert)
        vis_lt_r = p.createVisualShape(p.GEOM_CYLINDER, radius=0.022, length=0.05, rgbaColor=[0.95, 0.20, 0.20, 0.95])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vis_lt_r, basePosition=[-0.45, -0.95, 1.42 + fl_z])

        # 6. Drop Tray Outfeed Chute / Staging Ramp
        vis_chute = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.12, 0.18, 0.015], rgbaColor=[0.22, 0.24, 0.28, 1.0])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vis_chute, basePosition=[0.45, 0.65, 0.10])

    def _draw_workspace_bounds(self):
        """Draws crisp accent border lines around active picking workspace and safety perimeter."""
        x_lo, x_hi = 0.35, 0.65
        y_lo, y_hi = -0.20, 0.20
        z = 0.002
        accent_color = [0.08, 0.88, 0.92]  # Crisp Electric Cyan
        line_w = 2.5
        p.addUserDebugLine([x_lo, y_lo, z], [x_hi, y_lo, z], lineColorRGB=accent_color, lineWidth=line_w, lifeTime=0)
        p.addUserDebugLine([x_hi, y_lo, z], [x_hi, y_hi, z], lineColorRGB=accent_color, lineWidth=line_w, lifeTime=0)
        p.addUserDebugLine([x_hi, y_hi, z], [x_lo, y_hi, z], lineColorRGB=accent_color, lineWidth=line_w, lifeTime=0)
        p.addUserDebugLine([x_lo, y_hi, z], [x_lo, y_lo, z], lineColorRGB=accent_color, lineWidth=line_w, lifeTime=0)

        # Workcell Safety Perimeter (Yellow caution boundary matching ISO 10218 standard)
        px, py = 0.35, 0.12
        hx, hy = 0.85, 0.70
        z_floor = -0.620  # Slightly above floor
        yellow = [0.95, 0.78, 0.10]
        p.addUserDebugLine([px - hx, py - hy, z_floor], [px + hx, py - hy, z_floor], lineColorRGB=yellow, lineWidth=3.0, lifeTime=0)
        p.addUserDebugLine([px + hx, py - hy, z_floor], [px + hx, py + hy, z_floor], lineColorRGB=yellow, lineWidth=3.0, lifeTime=0)
        p.addUserDebugLine([px + hx, py + hy, z_floor], [px - hx, py + hy, z_floor], lineColorRGB=yellow, lineWidth=3.0, lifeTime=0)
        p.addUserDebugLine([px - hx, py + hy, z_floor], [px - hx, py - hy, z_floor], lineColorRGB=yellow, lineWidth=3.0, lifeTime=0)
        p.addUserDebugText("SAFETY WORKCELL PERIMETER", [px, py - hy - 0.08, z_floor + 0.01], textColorRGB=yellow, textSize=0.85, lifeTime=0)

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
