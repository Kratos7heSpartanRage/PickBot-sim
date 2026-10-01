import pybullet as p
import pybullet_data
import numpy as np
import time

from .camera import OverheadCamera
from .object_spawner import ObjectSpawner
from .robot import FrankaPandaRobot


class GraspingEnvironment:
    """
    PyBullet Simulated Picking Environment.
    Enhanced with better lighting, textures, workspace markers, and tray.
    """
    def __init__(self, gui=False):
        self.gui = gui
        connection_mode = p.GUI if self.gui else p.DIRECT
        self.client_id = p.connect(connection_mode)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / 240.0)

        if self.gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=1.5,
                cameraYaw=50.0,
                cameraPitch=-40.0,
                cameraTargetPosition=[0.45, 0.0, 0.15]
            )
            # Enable GUI panel for sliders and buttons
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)

        # Load environment assets
        self.plane_id = p.loadURDF("plane.urdf", [0, 0, -0.65])
        self.table_id = p.loadURDF("table/table.urdf", [0.5, 0.0, -0.65])

        # Add a tray on the side for collecting objects
        try:
            self.tray_id = p.loadURDF("tray/traybox.urdf", [0.5, 0.35, 0.0], globalScaling=0.6)
        except Exception:
            self.tray_id = None

        # Warm table surface color
        p.changeVisualShape(self.table_id, -1, rgbaColor=[0.6, 0.55, 0.5, 1.0])

        # Draw workspace boundaries in GUI
        if self.gui:
            self._draw_workspace_bounds()

        # Initialize robot arm
        self.robot = FrankaPandaRobot(base_pos=(0.0, 0.0, 0.0))

        # Initialize overhead camera
        self.camera = OverheadCamera(
            target_pos=(0.5, 0.0, 0.0),
            camera_pos=(0.5, 0.0, 0.85),
            img_size=(300, 300)
        )

        # Initialize object spawner on table
        self.spawner = ObjectSpawner(
            workspace_bounds=((0.38, 0.62), (-0.18, 0.18)),
            table_height=0.0
        )

    def _draw_workspace_bounds(self):
        """Draw visual workspace boundary markers in the simulation."""
        x_lo, x_hi = 0.38, 0.62
        y_lo, y_hi = -0.18, 0.18
        z = 0.002
        color = [0.2, 0.8, 0.2]
        width = 2.0
        p.addUserDebugLine([x_lo, y_lo, z], [x_hi, y_lo, z], lineColorRGB=color, lineWidth=width, lifeTime=0)
        p.addUserDebugLine([x_hi, y_lo, z], [x_hi, y_hi, z], lineColorRGB=color, lineWidth=width, lifeTime=0)
        p.addUserDebugLine([x_hi, y_hi, z], [x_lo, y_hi, z], lineColorRGB=color, lineWidth=width, lifeTime=0)
        p.addUserDebugLine([x_lo, y_hi, z], [x_lo, y_lo, z], lineColorRGB=color, lineWidth=width, lifeTime=0)
        p.addUserDebugText("WORKSPACE", [0.50, -0.21, z + 0.01], textColorRGB=[0.2, 0.8, 0.2], textSize=1.0)

    def reset(self, num_objects=3):
        """Resets robot to home and spawns randomized objects."""
        self.robot.reset()
        spawned_ids = self.spawner.spawn_random_objects(num_objects=num_objects)
        return spawned_ids

    def get_observation(self):
        """Captures overhead RGB, metric Depth, and segmentation mask."""
        return self.camera.capture()

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
