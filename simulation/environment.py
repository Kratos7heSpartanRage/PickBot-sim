import pybullet as p
import pybullet_data
import numpy as np

from .camera import OverheadCamera
from .object_spawner import ObjectSpawner
from .robot import FrankaPandaRobot


# Scene colour palettes (table, floor, workspace mat, accent lines, GUI background)
THEMES = {
    "cyber": {
        "floor": [0.06, 0.07, 0.10, 1], "table": [0.14, 0.15, 0.19, 1],
        "mat": [0.10, 0.12, 0.17, 1], "tray": [0.20, 0.23, 0.30, 1],
        "accent": [0.0, 0.85, 1.0], "bg": (0.04, 0.05, 0.08),
    },
    "industrial": {
        "floor": [0.30, 0.31, 0.33, 1], "table": [0.42, 0.43, 0.45, 1],
        "mat": [0.16, 0.30, 0.22, 1], "tray": [0.85, 0.55, 0.10, 1],
        "accent": [1.0, 0.75, 0.0], "bg": (0.22, 0.24, 0.27),
    },
    "studio": {
        "floor": [0.85, 0.86, 0.88, 1], "table": [0.96, 0.96, 0.97, 1],
        "mat": [0.80, 0.84, 0.90, 1], "tray": [0.35, 0.40, 0.48, 1],
        "accent": [0.15, 0.45, 0.95], "bg": (0.78, 0.80, 0.84),
    },
    "nordic": {
        "floor": [0.55, 0.50, 0.44, 1], "table": [0.76, 0.66, 0.52, 1],
        "mat": [0.32, 0.38, 0.40, 1], "tray": [0.92, 0.92, 0.90, 1],
        "accent": [0.95, 0.95, 0.95], "bg": (0.62, 0.66, 0.70),
    },
}
THEME_NAMES = list(THEMES.keys())


class GraspingEnvironment:
    """
    PyBullet simulation environment for robotic grasping.

    Coordinate convention: the TABLE TOP IS z = 0.0 (measured from the table
    AABB after loading). Previously the table was loaded 2.4 cm too low, so
    every object rested at z = -0.024 while the rest of the pipeline (spawner,
    detector table depth, grasp heights) assumed the table was at z = 0.
    """
    TABLE_URDF_TOP = 0.626     # table/table.urdf: top surface height above its base origin
    WORKSPACE = ((0.35, 0.65), (-0.20, 0.20))
    CAMERA_HEIGHT = 0.85

    def __init__(self, gui=False, theme="cyber", realtime=None, **kwargs):
        self.gui = gui
        self.theme_name = theme if theme in THEMES else "cyber"
        th = THEMES[self.theme_name]

        if self.gui:
            bg = th["bg"]
            opts = (f"--background_color_red={bg[0]} --background_color_green={bg[1]} "
                    f"--background_color_blue={bg[2]} --width=1400 --height=860")
            self.client_id = p.connect(p.GUI, options=opts)
        else:
            self.client_id = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / 240.0)
        p.setPhysicsEngineParameter(numSolverIterations=150, enableConeFriction=1)

        if self.gui:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)   # faster scene build
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
            try:
                p.configureDebugVisualizer(lightPosition=[1.5, -1.2, 2.5], shadowMapResolution=4096,
                                           shadowMapWorldSize=3, shadowMapIntensity=0.35)
            except TypeError:
                p.configureDebugVisualizer(lightPosition=[1.5, -1.2, 2.5])
            p.resetDebugVisualizerCamera(
                cameraDistance=1.25, cameraYaw=55.0, cameraPitch=-32.0,
                cameraTargetPosition=[0.42, 0.05, 0.05])

        table_base_z = -self.TABLE_URDF_TOP
        self.plane_id = p.loadURDF("plane.urdf", [0, 0, table_base_z])
        self.table_id = p.loadURDF("table/table.urdf", [0.5, 0.0, table_base_z])
        # Ground truth: measure where the table top actually is
        self.table_z = float(p.getAABB(self.table_id)[1][2])
        p.changeDynamics(self.table_id, -1, lateralFriction=1.0)

        # Thin visual-only workspace mat (no collision → does not change physics)
        (x0, x1), (y0, y1) = self.WORKSPACE
        mat_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[(x1 - x0) / 2 + 0.03, (y1 - y0) / 2 + 0.03, 0.0005])
        self.mat_id = p.createMultiBody(0, -1, mat_vis, [(x0 + x1) / 2, (y0 + y1) / 2, self.table_z + 0.0006])

        # Collection tray: placed just outside the camera's field of view
        try:
            self.tray_id = p.loadURDF("tray/traybox.urdf", [0.45, 0.50, self.table_z],
                                      globalScaling=0.55, useFixedBase=True)
        except Exception:
            self.tray_id = None

        # Visual camera rig (so the overhead sensor is visible in the 3D view)
        self._build_camera_rig()

        # Robot (real-time pacing in GUI → smooth, watchable motion)
        rt = self.gui if realtime is None else realtime
        self.robot = FrankaPandaRobot(base_pos=(0.0, 0.0, self.table_z), realtime=rt)

        self.camera = OverheadCamera(
            target_pos=(0.5, 0.0, self.table_z),
            camera_pos=(0.5, 0.0, self.table_z + self.CAMERA_HEIGHT),
            img_size=(300, 300),
        )
        # Camera-to-table distance: the detector needs this to segment objects from the table
        self.table_depth = self.camera.camera_pos[2] - self.table_z

        self.spawner = ObjectSpawner(workspace_bounds=self.WORKSPACE, table_height=self.table_z)

        self.workspace_lines = []
        self.apply_theme(self.theme_name)
        if self.gui:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    # ------------------------------------------------------------------
    # Visuals
    # ------------------------------------------------------------------
    def _build_camera_rig(self):
        cz = self.table_z + self.CAMERA_HEIGHT
        body = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.035, 0.025, 0.018], rgbaColor=[0.08, 0.08, 0.09, 1])
        lens = p.createVisualShape(p.GEOM_CYLINDER, radius=0.012, length=0.012, rgbaColor=[0.1, 0.5, 0.9, 1])
        self.cam_body_id = p.createMultiBody(0, -1, body, [0.5, 0.0, cz + 0.02])
        self.cam_lens_id = p.createMultiBody(0, -1, lens, [0.5, 0.0, cz + 0.002])
        # Gantry arm holding the camera (visual only, out of the camera's own view)
        beam = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.012, 0.36, 0.012], rgbaColor=[0.25, 0.27, 0.30, 1])
        floor_z = -self.TABLE_URDF_TOP
        post = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.015, 0.015, (cz + 0.06 - floor_z) / 2],
                                   rgbaColor=[0.25, 0.27, 0.30, 1])
        self.beam_id = p.createMultiBody(0, -1, beam, [0.5, -0.36, cz + 0.05])
        self.post_id = p.createMultiBody(0, -1, post, [0.5, -0.72, (cz + 0.06 + floor_z) / 2])

    def apply_theme(self, name):
        """Recolours the static scene. Objects keep their identity colours (matching their labels)."""
        if name not in THEMES:
            return
        self.theme_name = name
        th = THEMES[name]
        p.changeVisualShape(self.plane_id, -1, rgbaColor=th["floor"])
        p.changeVisualShape(self.table_id, -1, rgbaColor=th["table"], specularColor=[0.1, 0.1, 0.1])
        p.changeVisualShape(self.mat_id, -1, rgbaColor=th["mat"])
        if self.tray_id is not None:
            p.changeVisualShape(self.tray_id, -1, rgbaColor=th["tray"])
        if self.gui:
            self._draw_workspace(th["accent"])

    def _draw_workspace(self, color):
        for lid in self.workspace_lines:
            try:
                p.removeUserDebugItem(lid)
            except Exception:
                pass
        self.workspace_lines = []
        (x0, x1), (y0, y1) = self.WORKSPACE
        z = self.table_z + 0.002
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        for i in range(4):
            a, b = corners[i], corners[(i + 1) % 4]
            self.workspace_lines.append(p.addUserDebugLine([a[0], a[1], z], [b[0], b[1], z],
                                                           lineColorRGB=color, lineWidth=2.0, lifeTime=0))
        self.workspace_lines.append(p.addUserDebugText("WORKSPACE", [0.50, y0 - 0.04, z + 0.003],
                                                       textColorRGB=color, textSize=0.9))
        if self.tray_id is not None:
            self.workspace_lines.append(p.addUserDebugText("DROP TRAY", [0.45, 0.50, self.table_z + 0.10],
                                                           textColorRGB=color, textSize=0.9))

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------
    def reset(self, num_objects=3, **kwargs):
        """Resets robot and spawns objects."""
        self.robot.reset()
        return self.spawner.spawn_random_objects(num_objects=num_objects)

    def robot_pixels_in_view(self, seg):
        return int(np.count_nonzero(seg == self.robot.robot_id))

    def get_observation(self, ensure_clear_view=True):
        """
        Captures a FRESH overhead RGB, metric depth and segmentation frame.
        If the robot arm is inside the camera frustum it is first moved
        (smoothly) to its home pose, otherwise the CNN would detect grasps on
        the robot itself instead of the objects.
        """
        p.stepSimulation()
        rgb, depth, seg = self.camera.capture()
        if ensure_clear_view and self.robot_pixels_in_view(seg) > 0:
            self.robot.go_home()
            self.robot.step(10)
            rgb, depth, seg = self.camera.capture()
        return rgb, depth, seg

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
