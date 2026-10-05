import pybullet as p
import pybullet_data
import numpy as np

from .camera import OverheadCamera
from .object_spawner import ObjectSpawner
from .robot import FrankaPandaRobot


class GraspingEnvironment:
    """
    Clean PyBullet simulation environment for robotic grasping.
    Dark industrial table, simple workspace boundary, no text clutter.
    """
    def __init__(self, gui=False, **kwargs):
        self.gui = gui
        connection_mode = p.GUI if self.gui else p.DIRECT
        self.client_id = p.connect(connection_mode)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / 240.0)

        if self.gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=1.4,
                cameraYaw=50.0,
                cameraPitch=-35.0,
                cameraTargetPosition=[0.45, 0.0, 0.12]
            )
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)

        # Dark floor
        self.plane_id = p.loadURDF("plane.urdf", [0, 0, -0.65])
        p.changeVisualShape(self.plane_id, -1, rgbaColor=[0.20, 0.22, 0.25, 1.0])

        # Dark matte table
        self.table_id = p.loadURDF("table/table.urdf", [0.5, 0.0, -0.65])
        p.changeVisualShape(self.table_id, -1, rgbaColor=[0.18, 0.19, 0.22, 1.0])

        # Collection tray
        try:
            self.tray_id = p.loadURDF("tray/traybox.urdf", [0.5, 0.40, 0.0], globalScaling=0.55)
            p.changeVisualShape(self.tray_id, -1, rgbaColor=[0.28, 0.30, 0.33, 1.0])
        except Exception:
            self.tray_id = None

        # Clean workspace boundary
        if self.gui:
            self._draw_workspace()

        # Robot arm
        self.robot = FrankaPandaRobot(base_pos=(0.0, 0.0, 0.0))

        # Overhead camera
        self.camera = OverheadCamera(
            target_pos=(0.5, 0.0, 0.0),
            camera_pos=(0.5, 0.0, 0.85),
            img_size=(300, 300)
        )

        # Object spawner
        self.spawner = ObjectSpawner(
            workspace_bounds=((0.35, 0.65), (-0.20, 0.20)),
            table_height=0.0
        )

    def _draw_workspace(self):
        """Draw a clean, thin workspace boundary rectangle."""
        x0, x1 = 0.35, 0.65
        y0, y1 = -0.20, 0.20
        z = 0.002
        color = [0.0, 0.70, 0.90]
        w = 2.0

        p.addUserDebugLine([x0, y0, z], [x1, y0, z], lineColorRGB=color, lineWidth=w, lifeTime=0)
        p.addUserDebugLine([x1, y0, z], [x1, y1, z], lineColorRGB=color, lineWidth=w, lifeTime=0)
        p.addUserDebugLine([x1, y1, z], [x0, y1, z], lineColorRGB=color, lineWidth=w, lifeTime=0)
        p.addUserDebugLine([x0, y1, z], [x0, y0, z], lineColorRGB=color, lineWidth=w, lifeTime=0)

        p.addUserDebugText(
            "WORKSPACE",
            [0.50, -0.23, z + 0.003],
            textColorRGB=[0.0, 0.60, 0.80],
            textSize=0.9
        )

    def reset(self, num_objects=3, **kwargs):
        """Resets robot and spawns objects."""
        self.robot.reset()
        return self.spawner.spawn_random_objects(num_objects=num_objects)

    def get_observation(self):
        """Captures overhead RGB, metric Depth, and segmentation mask."""
        return self.camera.capture()

    def get_ground_truth_grasps(self):
        """Returns ground-truth Grasp objects for spawned objects."""
        return self.spawner.get_ground_truth_grasps(self.camera)

    def step(self):
        p.stepSimulation()

    def close(self):
        try:
            p.disconnect(self.client_id)
        except Exception:
            pass
