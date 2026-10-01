import numpy as np
import pybullet as p
import pybullet_data
import random
import os


class ObjectSpawner:
    """
    Advanced object spawner with diverse PyBullet URDF objects.
    Spawns a rich mix of shapes: lego bricks, mugs, dominos, jenga blocks,
    ducks, teddy bears, soccerballs, random mesh objects, plus procedural
    primitives in vibrant colors.
    """
    def __init__(self, workspace_bounds=((0.35, 0.65), (-0.2, 0.2)), table_height=0.0):
        self.x_bounds, self.y_bounds = workspace_bounds
        self.table_height = table_height
        self.spawned_ids = []
        self.object_data = []
        self.data_path = pybullet_data.getDataPath()

        self.palette = [
            [0.92, 0.22, 0.20, 1.0],   # Coral Red
            [0.20, 0.50, 0.90, 1.0],   # Ocean Blue
            [0.18, 0.80, 0.35, 1.0],   # Neon Green
            [0.95, 0.75, 0.10, 1.0],   # Sunflower Gold
            [0.72, 0.20, 0.90, 1.0],   # Vivid Purple
            [0.10, 0.88, 0.88, 1.0],   # Electric Cyan
            [0.98, 0.50, 0.12, 1.0],   # Tangerine
            [0.95, 0.30, 0.65, 1.0],   # Hot Pink
            [0.40, 0.85, 0.50, 1.0],   # Mint
            [0.60, 0.40, 0.20, 1.0],   # Warm Brown
        ]

        self.urdf_catalog = self._build_catalog()

    def _build_catalog(self):
        """Builds catalog of available URDF files from pybullet_data."""
        catalog = []
        named_objects = {
            "lego":       ("lego/lego.urdf",      0.8,   "mesh"),
            "domino":     ("domino/domino.urdf",   1.0,   "mesh"),
            "jenga":      ("jenga/jenga.urdf",     0.7,   "mesh"),
            "mug":        ("objects/mug.urdf",     0.12,  "mesh"),
            "duck":       ("duck_vhacd.urdf",      0.04,  "mesh"),
            "teddy":      ("teddy_vhacd.urdf",     0.015, "mesh"),
            "soccerball": ("soccerball.urdf",      0.25,  "mesh"),
            "cube_small": ("cube_small.urdf",      1.0,   "mesh"),
            "block":      ("block.urdf",           1.0,   "mesh"),
        }
        for name, (urdf_rel, scale, otype) in named_objects.items():
            urdf_path = os.path.join(self.data_path, urdf_rel)
            if os.path.exists(urdf_path):
                catalog.append({"name": name, "urdf": urdf_path, "scale": scale, "type": otype})

        random_urdfs_dir = os.path.join(self.data_path, "random_urdfs")
        if os.path.exists(random_urdfs_dir):
            indices = list(range(0, 1000, 33))[:30]
            for idx in indices:
                folder = os.path.join(random_urdfs_dir, f"{idx:03d}")
                urdf = os.path.join(folder, f"{idx:03d}.urdf")
                if os.path.exists(urdf):
                    catalog.append({"name": f"random_{idx:03d}", "urdf": urdf, "scale": 0.6, "type": "random_mesh"})
        return catalog

    def clear(self):
        for obj_id in self.spawned_ids:
            try:
                p.removeBody(obj_id)
            except Exception:
                pass
        self.spawned_ids.clear()
        self.object_data.clear()

    def _random_position(self, existing_positions, min_sep=0.08):
        """Generate a random position ensuring minimum separation."""
        for attempt in range(50):
            x = np.random.uniform(self.x_bounds[0] + 0.04, self.x_bounds[1] - 0.04)
            y = np.random.uniform(self.y_bounds[0] + 0.04, self.y_bounds[1] - 0.04)
            too_close = False
            for ex, ey in existing_positions:
                if np.sqrt((x - ex)**2 + (y - ey)**2) < min_sep:
                    too_close = True
                    break
            if not too_close:
                return x, y
        return (np.random.uniform(self.x_bounds[0] + 0.04, self.x_bounds[1] - 0.04),
                np.random.uniform(self.y_bounds[0] + 0.04, self.y_bounds[1] - 0.04))

    def spawn_random_objects(self, num_objects=3):
        """Spawns a diverse mix of URDF mesh objects and procedural primitives."""
        self.clear()
        existing_positions = []
        colors = random.sample(self.palette, min(num_objects, len(self.palette)))
        if num_objects > len(self.palette):
            colors = (colors * (num_objects // len(self.palette) + 1))[:num_objects]

        for i in range(num_objects):
            color = colors[i % len(colors)]
            x, y = self._random_position(existing_positions)
            existing_positions.append((x, y))
            z = self.table_height + 0.05
            yaw = np.random.uniform(-np.pi, np.pi)
            orientation = p.getQuaternionFromEuler([0, 0, yaw])

            use_urdf = (random.random() < 0.5) and len(self.urdf_catalog) > 0
            body_id = None
            obj_type = "primitive"
            dims = (0.04, 0.04, 0.03)
            obj_name = "primitive"

            if use_urdf:
                entry = random.choice(self.urdf_catalog)
                obj_name = entry["name"]
                obj_type = entry["type"]
                try:
                    scale = entry["scale"]
                    if obj_name == "soccerball":
                        scale, z = 0.25, self.table_height + 0.03
                    elif obj_name == "duck":
                        scale, z = 0.04, self.table_height + 0.02
                    elif obj_name == "teddy":
                        scale, z = 0.015, self.table_height + 0.02
                    elif obj_name == "mug":
                        scale, z = 0.12, self.table_height + 0.02
                    elif obj_name.startswith("random_"):
                        scale, z = np.random.uniform(0.4, 0.8), self.table_height + 0.04

                    body_id = p.loadURDF(entry["urdf"], basePosition=[x, y, z],
                                         baseOrientation=orientation, globalScaling=scale, useFixedBase=False)
                    p.changeVisualShape(body_id, -1, rgbaColor=color)
                    for link_idx in range(p.getNumJoints(body_id)):
                        p.changeVisualShape(body_id, link_idx, rgbaColor=color)
                    dims = (0.04, 0.04, 0.03)
                except Exception:
                    body_id = None

            if body_id is None:
                prim_type = np.random.choice(["box", "cylinder", "bar", "sphere", "capsule"])
                obj_name = prim_type
                if prim_type == "box":
                    sx, sy, sz = np.random.uniform(0.025, 0.05), np.random.uniform(0.025, 0.05), np.random.uniform(0.018, 0.035)
                    col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[sx/2, sy/2, sz/2])
                    vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[sx/2, sy/2, sz/2], rgbaColor=color)
                    dims = (sx, sy, sz)
                elif prim_type == "bar":
                    sx, sy, sz = np.random.uniform(0.06, 0.10), np.random.uniform(0.022, 0.032), np.random.uniform(0.015, 0.025)
                    col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[sx/2, sy/2, sz/2])
                    vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[sx/2, sy/2, sz/2], rgbaColor=color)
                    dims = (sx, sy, sz)
                elif prim_type == "cylinder":
                    radius, length = np.random.uniform(0.015, 0.03), np.random.uniform(0.03, 0.06)
                    col_id = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=length)
                    vis_id = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=length, rgbaColor=color)
                    dims = (radius*2, radius*2, length)
                elif prim_type == "sphere":
                    radius = np.random.uniform(0.015, 0.03)
                    col_id = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
                    vis_id = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
                    dims = (radius*2, radius*2, radius*2)
                else:
                    radius, length = np.random.uniform(0.012, 0.02), np.random.uniform(0.04, 0.07)
                    col_id = p.createCollisionShape(p.GEOM_CAPSULE, radius=radius, height=length)
                    vis_id = p.createVisualShape(p.GEOM_CAPSULE, radius=radius, length=length, rgbaColor=color)
                    dims = (radius*2, radius*2, length)
                body_id = p.createMultiBody(baseMass=0.15, baseCollisionShapeIndex=col_id, baseVisualShapeIndex=vis_id,
                                            basePosition=[x, y, z], baseOrientation=orientation)
                obj_type = prim_type

            p.changeDynamics(body_id, -1, lateralFriction=0.8, spinningFriction=0.003, rollingFriction=0.003, restitution=0.1, mass=0.15)
            self.spawned_ids.append(body_id)
            self.object_data.append({"id": body_id, "type": obj_type, "name": obj_name, "color": color, "init_pose": ([x, y, z], yaw), "dims": dims})

        for _ in range(120):
            p.stepSimulation()

        for data in self.object_data:
            try:
                pos, orn = p.getBasePositionAndOrientation(data["id"])
                euler = p.getEulerFromQuaternion(orn)
                data["settled_pos"] = np.array(pos)
                data["settled_yaw"] = euler[2]
            except Exception:
                data["settled_pos"] = np.array(data["init_pose"][0])
                data["settled_yaw"] = data["init_pose"][1]
        return self.spawned_ids

    def get_ground_truth_grasps(self, camera):
        """Generates ground-truth Grasp objects for all settled objects."""
        from perception.grasp_geometry import Grasp
        gt_grasps = []
        for data in self.object_data:
            pos = data["settled_pos"]
            yaw = data["settled_yaw"]
            u, v = camera.project_world_to_pixel(pos)
            z_cam = camera.camera_pos[2] - pos[2]
            f_x = camera.K[0, 0]
            meter_width = data["dims"][1] if data["type"] == "bar" else data["dims"][0]
            width_px = max(15.0, (meter_width * f_x) / z_cam)
            gt_grasps.append(Grasp(center_px=(u, v), angle_rad=yaw, width_px=width_px, quality=1.0, world_coords=pos))
        return gt_grasps

    def get_object_info(self):
        """Returns a list of human-readable info about spawned objects."""
        info = []
        for data in self.object_data:
            pos = data.get("settled_pos", data["init_pose"][0])
            info.append({"name": data["name"], "type": data["type"],
                         "position": [round(float(v), 3) for v in pos],
                         "color_rgb": [round(c, 2) for c in data["color"][:3]]})
        return info
