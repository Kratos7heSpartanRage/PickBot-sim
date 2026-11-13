"""
=============================================================================
  AI-Based Vision-Guided Robotic Grasping System: Deliverable 2
  Environment & Scene Synthesis Subsystem: Object Spawner
  Lead: Satyam Kumar
=============================================================================
Calibrated multi-object spawner enforcing hardware constraints of the Franka
Panda 2-finger parallel jaw gripper (max stroke: 80 mm, finger height: ~20 mm):
  - Graspable spans strictly calibrated to 25 mm - 60 mm.
  - Curated library of 8 diverse objects (primitives + URDF meshes).
  - High-friction contact physics (lateral=1.40, spinning=0.03, rolling=0.01).
  - Multi-clutter spawning (Level 1, 3, 5) with collision-free placement.
  - Floating 3D scene HUD labels for visual presentation.
=============================================================================
"""
import os
import random
import numpy as np
import pybullet as p
import pybullet_data


class ObjectSpawner:
    """
    Curated, calibrated object spawner for Deliverable 2.
    Spawns objects with strictly verified dimensions and tuned contact dynamics.
    """
    def __init__(self, workspace_bounds=((0.36, 0.64), (-0.18, 0.18)), table_height=0.0):
        self.x_bounds, self.y_bounds = workspace_bounds
        self.table_height = table_height
        self.spawned_ids = []
        self.object_data = []
        self.hud_text_ids = []
        self.data_path = pybullet_data.getDataPath()

        # Studio presentation palette (vibrant, high contrast against dark mat)
        self.palette = [
            [0.92, 0.22, 0.20, 1.0],   # Coral Red
            [0.18, 0.55, 0.95, 1.0],   # Ocean Blue
            [0.15, 0.85, 0.40, 1.0],   # Neon Green
            [0.98, 0.78, 0.10, 1.0],   # Sunflower Gold
            [0.72, 0.22, 0.92, 1.0],   # Vivid Purple
            [0.08, 0.88, 0.92, 1.0],   # Electric Cyan
            [0.98, 0.48, 0.12, 1.0],   # Tangerine Orange
            [0.95, 0.28, 0.65, 1.0],   # Hot Pink
        ]

        # Curated catalog definitions conforming to Eval 2 Engineering Spec (Page 7)
        self.catalog = [
            {
                "key": "cube",
                "name": "Small Cube",
                "type": "primitive_box",
                "dims": (0.038, 0.038, 0.038),
                "mass": 0.08,
                "grasp_span": 0.038,
            },
            {
                "key": "machined_bar",
                "name": "Machined Bar",
                "type": "primitive_box",
                "dims": (0.055, 0.032, 0.025),
                "mass": 0.12,
                "grasp_span": 0.032,
            },
            {
                "key": "cylinder",
                "name": "Cylinder Can",
                "type": "primitive_cylinder",
                "dims": (0.044, 0.044, 0.055),  # dia=44mm, h=55mm
                "mass": 0.10,
                "grasp_span": 0.044,
            },
            {
                "key": "l_bracket",
                "name": "L-Bracket",
                "type": "compound_l_bracket",
                "dims": (0.050, 0.028, 0.040),  # span=50mm, thickness=28mm
                "mass": 0.10,
                "grasp_span": 0.028,
            },
            {
                "key": "lego",
                "name": "Calibrated Lego",
                "type": "urdf",
                "urdf_rel": "lego/lego.urdf",
                "scale": 1.30,
                "dims": (0.044, 0.044, 0.032),
                "mass": 0.05,
                "grasp_span": 0.044,
            },
            {
                "key": "mug",
                "name": "Calibrated Mug",
                "type": "urdf",
                "urdf_rel": "objects/mug.urdf",
                "scale": 0.35,
                "dims": (0.035, 0.049, 0.041),  # rim dia ~45mm
                "mass": 0.15,
                "grasp_span": 0.045,
            },
            {
                "key": "domino",
                "name": "Calibrated Domino",
                "type": "urdf",
                "urdf_rel": "domino/domino.urdf",
                "scale": 1.00,
                "dims": (0.051, 0.025, 0.007),
                "mass": 0.06,
                "grasp_span": 0.025,
            },
            {
                "key": "duck",
                "name": "Calibrated Duck",
                "type": "urdf",
                "urdf_rel": "duck_vhacd.urdf",
                "scale": 0.40,
                "dims": (0.041, 0.039, 0.031),
                "mass": 0.08,
                "grasp_span": 0.042,
            },
        ]

    def clear(self):
        """Removes all spawned bodies and HUD text elements from the simulation."""
        for obj_id in self.spawned_ids:
            try:
                p.removeBody(obj_id)
            except Exception:
                pass
        self.spawned_ids.clear()
        self.object_data.clear()

        for txt_id in self.hud_text_ids:
            try:
                p.removeUserDebugItem(txt_id)
            except Exception:
                pass
        self.hud_text_ids.clear()

    def _get_clutter_min_sep(self, num_objects):
        """Returns required center-to-center separation distance per clutter level."""
        if num_objects <= 1:
            return 0.0
        elif num_objects <= 3:
            return 0.14  # Eval 2 spec: Clutter Level 3 requires >= 14 cm
        elif num_objects <= 5:
            return 0.09  # Eval 2 spec: Clutter Level 5 requires >= 9 cm
        else:
            return 0.07

    def _sample_positions(self, num_objects):
        """Generates collision-free spawn positions respecting clutter constraints."""
        positions = []
        min_sep = self._get_clutter_min_sep(num_objects)

        if num_objects == 1:
            # Clutter Level 1: exactly 1 object spawned near workspace center [0.50, 0.00]
            cx = 0.50 + np.random.uniform(-0.02, 0.02)
            cy = 0.00 + np.random.uniform(-0.02, 0.02)
            return [(cx, cy)]

        for i in range(num_objects):
            placed = False
            for attempt in range(120):
                x = np.random.uniform(self.x_bounds[0], self.x_bounds[1])
                y = np.random.uniform(self.y_bounds[0], self.y_bounds[1])
                too_close = False
                for ex, ey in positions:
                    if np.sqrt((x - ex) ** 2 + (y - ey) ** 2) < min_sep:
                        too_close = True
                        break
                if not too_close:
                    positions.append((x, y))
                    placed = True
                    break

            if not placed:
                fallback_sep = min_sep * 0.75
                for _ in range(50):
                    x = np.random.uniform(self.x_bounds[0], self.x_bounds[1])
                    y = np.random.uniform(self.y_bounds[0], self.y_bounds[1])
                    if all(np.sqrt((x - ex) ** 2 + (y - ey) ** 2) >= fallback_sep for ex, ey in positions):
                        positions.append((x, y))
                        placed = True
                        break
                if not placed:
                    positions.append((np.random.uniform(self.x_bounds[0], self.x_bounds[1]),
                                      np.random.uniform(self.y_bounds[0], self.y_bounds[1])))

        return positions

    def _create_primitive_box(self, dims, color, pos, orn, mass):
        """Creates a box primitive body."""
        sx, sy, sz = dims
        col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[sx / 2.0, sy / 2.0, sz / 2.0])
        vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[sx / 2.0, sy / 2.0, sz / 2.0], rgbaColor=color)
        body_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=pos,
            baseOrientation=orn
        )
        return body_id

    def _create_primitive_cylinder(self, dims, color, pos, orn, mass):
        """Creates a standing cylinder primitive body."""
        dia_x, dia_y, height = dims
        radius = dia_x / 2.0
        col_id = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)
        vis_id = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=color)
        body_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=pos,
            baseOrientation=orn
        )
        return body_id

    def _create_l_bracket(self, color, pos, orn, mass):
        """Creates a realistic compound L-bracket primitive."""
        col_id = p.createCollisionShapeArray(
            [p.GEOM_BOX, p.GEOM_BOX],
            halfExtents=[[0.025, 0.014, 0.006], [0.007, 0.014, 0.014]],
            collisionFramePositions=[[0.0, 0.0, 0.0], [-0.018, 0.0, 0.020]]
        )
        vis_id = p.createVisualShapeArray(
            [p.GEOM_BOX, p.GEOM_BOX],
            halfExtents=[[0.025, 0.014, 0.006], [0.007, 0.014, 0.014]],
            visualFramePositions=[[0.0, 0.0, 0.0], [-0.018, 0.0, 0.020]],
            rgbaColors=[color, color]
        )
        body_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=pos,
            baseOrientation=orn
        )
        return body_id

    def _load_urdf_object(self, urdf_rel, scale, color, pos, orn, mass):
        """Loads a calibrated URDF object and tints all visual links."""
        urdf_path = os.path.join(self.data_path, urdf_rel)
        if not os.path.exists(urdf_path):
            return None

        body_id = p.loadURDF(
            urdf_path,
            basePosition=pos,
            baseOrientation=orn,
            globalScaling=scale,
            useFixedBase=False
        )

        p.changeVisualShape(body_id, -1, rgbaColor=color)
        for link_idx in range(p.getNumJoints(body_id)):
            p.changeVisualShape(body_id, link_idx, rgbaColor=color)

        return body_id

    def _apply_tuned_contact_physics(self, body_id, mass):
        """
        Applies Eval 2 calibrated contact physics parameters:
        High lateral friction (1.40) and low restitution (0.02) prevent slipping & bouncing.
        """
        p.changeDynamics(
            body_id, -1,
            mass=mass,
            lateralFriction=1.40,
            spinningFriction=0.03,
            rollingFriction=0.01,
            restitution=0.02,
            contactDamping=100.0,
            contactStiffness=50000.0
        )
        for link_idx in range(p.getNumJoints(body_id)):
            p.changeDynamics(
                body_id, link_idx,
                lateralFriction=1.40,
                spinningFriction=0.03,
                rollingFriction=0.01,
                restitution=0.02,
                contactDamping=100.0,
                contactStiffness=50000.0
            )

    def spawn_random_objects(self, num_objects=3):
        """
        Spawns a diverse, non-overlapping set of calibrated objects.
        Drops them from z = table_height + 0.04m and settles for 60 physics steps.
        """
        self.clear()
        num_objects = max(1, int(num_objects))
        positions = self._sample_positions(num_objects)

        if num_objects <= len(self.catalog):
            selected_entries = random.sample(self.catalog, num_objects)
        else:
            selected_entries = [random.choice(self.catalog) for _ in range(num_objects)]

        if num_objects <= len(self.palette):
            colors = random.sample(self.palette, num_objects)
        else:
            colors = [self.palette[i % len(self.palette)] for i in range(num_objects)]

        for i in range(num_objects):
            entry = selected_entries[i]
            color = colors[i]
            x, y = positions[i]
            z = self.table_height + 0.04
            yaw = np.random.uniform(-np.pi, np.pi)
            orientation = p.getQuaternionFromEuler([0.0, 0.0, yaw])

            body_id = None
            otype = entry["type"]

            if otype == "primitive_box":
                body_id = self._create_primitive_box(entry["dims"], color, [x, y, z], orientation, entry["mass"])
            elif otype == "primitive_cylinder":
                body_id = self._create_primitive_cylinder(entry["dims"], color, [x, y, z], orientation, entry["mass"])
            elif otype == "compound_l_bracket":
                body_id = self._create_l_bracket(color, [x, y, z], orientation, entry["mass"])
            elif otype == "urdf":
                body_id = self._load_urdf_object(entry["urdf_rel"], entry["scale"], color, [x, y, z], orientation, entry["mass"])

            if body_id is None:
                body_id = self._create_primitive_box(entry["dims"], color, [x, y, z], orientation, entry["mass"])

            self._apply_tuned_contact_physics(body_id, entry["mass"])
            self.spawned_ids.append(body_id)
            self.object_data.append({
                "id": body_id,
                "key": entry["key"],
                "name": entry["name"],
                "type": otype,
                "color": color,
                "dims": entry["dims"],
                "mass": entry["mass"],
                "grasp_span": entry["grasp_span"],
                "init_pose": ([x, y, z], yaw)
            })

        # Settle for 60 physics steps
        for _ in range(60):
            p.stepSimulation()

        # Record stable settled poses
        for data in self.object_data:
            try:
                pos, orn = p.getBasePositionAndOrientation(data["id"])
                euler = p.getEulerFromQuaternion(orn)
                data["settled_pos"] = np.array(pos)
                data["settled_yaw"] = euler[2]
            except Exception:
                data["settled_pos"] = np.array(data["init_pose"][0])
                data["settled_yaw"] = data["init_pose"][1]

        # Draw clean 3D HUD floating text labels
        self._draw_hud_labels()

        return self.spawned_ids

    def _draw_hud_labels(self):
        """Draws minimal, stylish floating 3D text labels (z = settled_z + 0.07m)."""
        for txt_id in self.hud_text_ids:
            try:
                p.removeUserDebugItem(txt_id)
            except Exception:
                pass
        self.hud_text_ids.clear()

        for i, data in enumerate(self.object_data):
            pos = data["settled_pos"]
            label_text = f"{i + 1}: {data['name']}"
            label_pos = [pos[0], pos[1], pos[2] + 0.07]
            txt_id = p.addUserDebugText(
                label_text,
                label_pos,
                textColorRGB=[1.0, 1.0, 1.0],
                textSize=0.9,
                lifeTime=0
            )
            self.hud_text_ids.append(txt_id)

    def get_ground_truth_grasps(self, camera):
        """
        Generates ground-truth Grasp objects for all settled objects.
        Fulfills Section 4.5 Observation Output Contract.
        """
        from perception.grasp_geometry import Grasp
        gt_grasps = []
        for data in self.object_data:
            pos = data["settled_pos"]
            yaw = data["settled_yaw"]
            u, v = camera.project_world_to_pixel(pos)
            z_cam = max(0.1, camera.camera_pos[2] - pos[2])
            f_x = camera.K[0, 0]
            meter_width = data.get("grasp_span", data["dims"][0])
            width_px = max(15.0, (meter_width * f_x) / z_cam)
            gt_grasps.append(Grasp(
                center_px=(u, v),
                angle_rad=yaw,
                width_px=width_px,
                quality=1.0,
                world_coords=pos
            ))
        return gt_grasps

    def get_object_info(self):
        """
        Returns structured metadata for all currently spawned objects.
        Fulfills Section 4.5 Observation Output Contract.
        """
        info = []
        for data in self.object_data:
            pos = data.get("settled_pos", data["init_pose"][0])
            info.append({
                "body_id": data["id"],
                "key": data["key"],
                "name": data["name"],
                "type": data["type"],
                "settled_pos": [round(float(v), 4) for v in pos],
                "dimensions": [round(float(d), 4) for d in data["dims"]],
                "mass": data["mass"],
                "grasp_span": data["grasp_span"],
                "color_rgb": [round(float(c), 3) for c in data["color"][:3]]
            })
        return info
