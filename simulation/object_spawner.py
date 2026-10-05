import numpy as np
import pybullet as p
import pybullet_data
import random
import os


class ObjectSpawner:
    """
    Simple Object Spawner with 5 clearly visible, well-defined dataset objects.
    Each object is a real URDF from PyBullet's built-in dataset, scaled large
    enough to be instantly recognizable in the 3D scene.
    """

    # 5 Simple, Recognizable Objects from PyBullet Dataset
    OBJECT_LIBRARY = [
        {
            "name": "Duck",
            "urdf": "duck_vhacd.urdf",
            "scale": 0.55,
            "spawn_z": 0.08,
            "color": [1.00, 0.85, 0.00, 1.0],   # Bright Yellow
            "default_dims": (0.06, 0.05, 0.06),
        },
        {
            "name": "Lego",
            "urdf": "lego/lego.urdf",
            "scale": 3.5,
            "spawn_z": 0.04,
            "color": [0.95, 0.15, 0.12, 1.0],   # Bright Red
            "default_dims": (0.10, 0.05, 0.035),
        },
        {
            "name": "Mug",
            "urdf": "objects/mug.urdf",
            "scale": 0.45,
            "spawn_z": 0.06,
            "color": [0.12, 0.50, 0.95, 1.0],   # Ocean Blue
            "default_dims": (0.06, 0.06, 0.07),
        },
        {
            "name": "Domino",
            "urdf": "domino/domino.urdf",
            "scale": 2.8,
            "spawn_z": 0.03,
            "color": [0.10, 0.90, 0.25, 1.0],   # Neon Green
            "default_dims": (0.10, 0.05, 0.018),
        },
        {
            "name": "Jenga",
            "urdf": "jenga/jenga.urdf",
            "scale": 2.0,
            "spawn_z": 0.04,
            "color": [0.75, 0.25, 0.95, 1.0],   # Electric Purple
            "default_dims": (0.12, 0.04, 0.030),
        },
    ]

    def __init__(self, workspace_bounds=((0.35, 0.65), (-0.20, 0.20)), table_height=0.0):
        self.x_bounds, self.y_bounds = workspace_bounds
        self.table_height = table_height
        self.spawned_ids = []
        self.object_data = []
        self.label_ids = []  # 3D text labels above objects
        self.data_path = pybullet_data.getDataPath()

    def clear(self):
        """Remove all spawned objects and labels."""
        for obj_id in self.spawned_ids:
            try:
                p.removeBody(obj_id)
            except Exception:
                pass
        for label_id in self.label_ids:
            try:
                p.removeUserDebugItem(label_id)
            except Exception:
                pass
        self.spawned_ids.clear()
        self.object_data.clear()
        self.label_ids.clear()

    def _random_position(self, existing_positions, min_sep=0.16):
        """Generate random position inside workspace with separation."""
        for _ in range(100):
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

    def spawn_random_objects(self, num_objects=3, **kwargs):
        """
        Spawns N objects chosen from the 5-object library.
        Each object is large, colorful, and clearly visible.
        """
        self.clear()
        num_objects = max(1, min(num_objects, 5))
        existing_positions = []

        # Shuffle and pick N objects from the library
        chosen = random.sample(self.OBJECT_LIBRARY, num_objects)

        for i, entry in enumerate(chosen):
            x, y = self._random_position(existing_positions)
            existing_positions.append((x, y))
            z = self.table_height + entry["spawn_z"]
            yaw = np.random.uniform(-np.pi, np.pi)
            orientation = p.getQuaternionFromEuler([0, 0, yaw])

            urdf_path = os.path.join(self.data_path, entry["urdf"])
            if not os.path.exists(urdf_path):
                continue

            body_id = p.loadURDF(
                urdf_path,
                basePosition=[x, y, z],
                baseOrientation=orientation,
                globalScaling=entry["scale"],
                useFixedBase=False
            )

            # Apply bright distinct color
            color = entry["color"]
            p.changeVisualShape(body_id, -1, rgbaColor=color)
            for link_idx in range(p.getNumJoints(body_id)):
                p.changeVisualShape(body_id, link_idx, rgbaColor=color)

            # High-grip physics for stable grasping
            p.changeDynamics(
                body_id, -1,
                lateralFriction=1.5,
                spinningFriction=0.03,
                rollingFriction=0.03,
                restitution=0.05,
                mass=0.20
            )

            self.spawned_ids.append(body_id)
            self.object_data.append({
                "id": body_id,
                "type": "urdf",
                "name": entry["name"],
                "color": color,
                "init_pose": ([x, y, z], yaw),
                "dims": entry["default_dims"],
            })

        # Let objects settle
        for _ in range(120):
            p.stepSimulation()

        # Record settled poses and add 3D labels
        for idx, data in enumerate(self.object_data):
            try:
                pos, orn = p.getBasePositionAndOrientation(data["id"])
                euler = p.getEulerFromQuaternion(orn)
                data["settled_pos"] = np.array(pos)
                data["settled_yaw"] = euler[2]

                # Compute actual bounding box dims
                aabb_min, aabb_max = p.getAABB(data["id"])
                data["dims"] = tuple(aabb_max[i] - aabb_min[i] for i in range(3))
            except Exception:
                data["settled_pos"] = np.array(data["init_pose"][0])
                data["settled_yaw"] = data["init_pose"][1]

            # Floating name label above object
            label_pos = list(data["settled_pos"])
            label_pos[2] += 0.07
            label_id = p.addUserDebugText(
                f"{idx + 1}: {data['name']}",
                label_pos,
                textColorRGB=data["color"][:3],
                textSize=1.3
            )
            self.label_ids.append(label_id)

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
            meter_width = data["dims"][0]
            width_px = max(18.0, (meter_width * f_x) / z_cam)
            gt_grasps.append(Grasp(
                center_px=(u, v),
                angle_rad=yaw,
                width_px=width_px,
                quality=1.0,
                world_coords=pos
            ))
        return gt_grasps

    def get_object_info(self):
        """Returns list of human-readable info about spawned objects."""
        info = []
        for data in self.object_data:
            pos = data.get("settled_pos", data["init_pose"][0])
            info.append({
                "name": data["name"],
                "type": data["type"],
                "position": [round(float(v), 3) for v in pos],
                "color_rgb": [round(c, 2) for c in data["color"][:3]],
            })
        return info
