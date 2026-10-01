import numpy as np
import pybullet as p

class OverheadCamera:
    """
    Simulated overhead RGB-D camera looking top-down at the picking workspace.
    Computes camera intrinsics/extrinsics and converts 2D pixel coordinates + depth
    into 3D Cartesian coordinates in world space.
    """
    def __init__(self,
                 target_pos=(0.5, 0.0, 0.0),
                 camera_pos=(0.5, 0.0, 0.85),
                 up_vector=(0, 1, 0),
                 img_size=(300, 300),
                 fov=45.0,
                 near_plane=0.1,
                 far_plane=1.5):
        self.target_pos = target_pos
        self.camera_pos = camera_pos
        self.up_vector = up_vector
        self.img_w, self.img_h = img_size
        self.fov = fov
        self.near = near_plane
        self.far = far_plane

        self.aspect = float(self.img_w) / float(self.img_h)
        self.view_matrix = p.computeViewMatrix(
            cameraEyePosition=self.camera_pos,
            cameraTargetPosition=self.target_pos,
            cameraUpVector=self.up_vector
        )
        self.proj_matrix = p.computeProjectionMatrixFOV(
            fov=self.fov,
            aspect=self.aspect,
            nearVal=self.near,
            farVal=self.far
        )

        # Precompute intrinsic matrix K for pinhole model
        # f = (height / 2) / tan(fov_y / 2)
        f_y = (self.img_h / 2.0) / np.tan(np.radians(self.fov) / 2.0)
        f_x = f_y  # Square pixels
        c_x = self.img_w / 2.0
        c_y = self.img_h / 2.0
        self.K = np.array([
            [f_x,  0.0, c_x],
            [0.0,  f_y, c_y],
            [0.0,  0.0, 1.0]
        ], dtype=float)

    def capture(self):
        """
        Captures RGB, linear metric Depth, and segmentation mask.
        """
        _, _, rgb_raw, depth_raw, seg_raw = p.getCameraImage(
            width=self.img_w,
            height=self.img_h,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )

        # RGB image [H, W, 3] in uint8
        rgb = np.array(rgb_raw, dtype=np.uint8).reshape((self.img_h, self.img_w, 4))[:, :, :3]

        # Convert OpenGL non-linear depth buffer [0, 1] into linear depth in meters
        depth_buffer = np.array(depth_raw, dtype=np.float32).reshape((self.img_h, self.img_w))
        # metric_depth = far * near / (far - (far - near) * depth_buffer)
        metric_depth = self.far * self.near / (self.far - (self.far - self.near) * depth_buffer)

        # Segmentation mask [H, W] int32
        seg = np.array(seg_raw, dtype=np.int32).reshape((self.img_h, self.img_w))

        return rgb, metric_depth, seg

    def deproject_pixel_to_world(self, u, v, depth_val):
        """
        Converts 2D pixel (u, v) and metric depth into 3D Cartesian coordinates in world space.
        """
        # Convert pixel to camera frame:
        # Z_c = depth_val
        # X_c = (u - c_x) * Z_c / f_x
        # Y_c = (v - c_y) * Z_c / f_y
        c_x = self.K[0, 2]
        c_y = self.K[1, 2]
        f_x = self.K[0, 0]
        f_y = self.K[1, 1]

        z_cam = depth_val
        x_cam = (u - c_x) * z_cam / f_x
        # In OpenGL/PyBullet camera conventions, Y goes downwards in image, upwards in cam
        y_cam = -(v - c_y) * z_cam / f_y

        # Camera frame to world frame transform:
        # With top-down camera at [0.5, 0.0, 0.85] and up=[0, 1, 0]:
        # camera X aligns with World X
        # camera Y aligns with World Y
        # camera Z is (camera_pos_z - depth_val)
        x_world = self.camera_pos[0] + x_cam
        y_world = self.camera_pos[1] + y_cam
        z_world = self.camera_pos[2] - z_cam

        return np.array([x_world, y_world, z_world], dtype=float)

    def project_world_to_pixel(self, world_coord):
        """
        Projects 3D world coordinate [x, y, z] to 2D pixel (u, v).
        """
        x_w, y_w, z_w = world_coord
        z_cam = self.camera_pos[2] - z_w
        x_cam = x_w - self.camera_pos[0]
        y_cam = y_w - self.camera_pos[1]

        c_x = self.K[0, 2]
        c_y = self.K[1, 2]
        f_x = self.K[0, 0]
        f_y = self.K[1, 1]

        u = (x_cam * f_x / z_cam) + c_x
        v = (-y_cam * f_y / z_cam) + c_y
        return int(round(u)), int(round(v))
