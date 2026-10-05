import numpy as np
import cv2

class Grasp:
    """
    Representation of an antipodal robotic grasp.
    Contains 2D pixel coordinates, angle, width, grasp quality,
    and corresponding 3D world coordinates.
    """
    def __init__(self, center_px, angle_rad, width_px, quality=1.0, world_coords=None):
        self.u, self.v = float(center_px[0]), float(center_px[1])
        self.angle_rad = float(angle_rad)
        self.angle_deg = float(np.degrees(self.angle_rad))
        self.width_px = float(width_px)
        self.quality = float(quality)
        self.world_coords = np.array(world_coords, dtype=float) if world_coords is not None else None

    @property
    def center(self):
        return np.array([self.u, self.v], dtype=float)

    def get_rect_points(self, jaw_length=20.0):
        """
        Returns 4 corner points of the oriented grasp rectangle in pixel space.
        jaw_length: length of gripper fingers in pixels.
        """
        c = np.cos(self.angle_rad)
        s = np.sin(self.angle_rad)

        # Gripper opening direction (across fingers)
        w_vec = np.array([c, s]) * (self.width_px / 2.0)
        # Gripper finger length direction (along finger)
        l_vec = np.array([-s, c]) * (jaw_length / 2.0)

        center = self.center
        p1 = center - w_vec - l_vec
        p2 = center + w_vec - l_vec
        p3 = center + w_vec + l_vec
        p4 = center - w_vec + l_vec

        return np.array([p1, p2, p3, p4], dtype=np.float32)

    def get_jaw_lines(self, jaw_length=20.0):
        """
        Returns the two parallel line segments representing the left and right gripper jaws.
        Each line is ((x1, y1), (x2, y2)).
        """
        c = np.cos(self.angle_rad)
        s = np.sin(self.angle_rad)

        w_vec = np.array([c, s]) * (self.width_px / 2.0)
        l_vec = np.array([-s, c]) * (jaw_length / 2.0)

        center = self.center
        left_jaw = (center - w_vec - l_vec, center - w_vec + l_vec)
        right_jaw = (center + w_vec - l_vec, center + w_vec + l_vec)
        center_line = (center - w_vec, center + w_vec)
        return left_jaw, right_jaw, center_line

    def compute_iou(self, other_grasp):
        """
        Computes Jaccard IoU between this grasp rectangle and another.
        Uses OpenCV rotated rectangle intersection.
        """
        rect1 = ((self.u, self.v), (self.width_px, 20.0), self.angle_deg)
        rect2 = ((other_grasp.u, other_grasp.v), (other_grasp.width_px, 20.0), other_grasp.angle_deg)

        try:
            ret, pts = cv2.rotatedRectangleIntersection(rect1, rect2)
            if ret == cv2.INTERSECT_NONE or pts is None:
                return 0.0
            inter_area = cv2.contourArea(pts)
            area1 = self.width_px * 20.0
            area2 = other_grasp.width_px * 20.0
            union_area = area1 + area2 - inter_area
            if union_area <= 0:
                return 0.0
            return inter_area / union_area
        except Exception:
            return 0.0

    def is_valid_match(self, gt_grasp, iou_thresh=0.25, angle_thresh_deg=30.0):
        """
        Cornell grasp evaluation criteria:
        1. Grasp rectangle IoU >= 0.25
        2. Orientation difference <= 30 degrees (modulo 180 degrees)
        """
        iou = self.compute_iou(gt_grasp)
        angle_diff = np.abs(self.angle_deg - gt_grasp.angle_deg) % 180.0
        if angle_diff > 90.0:
            angle_diff = 180.0 - angle_diff
        return (iou >= iou_thresh) and (angle_diff <= angle_thresh_deg)

    @property
    def gripper_yaw(self):
        """Yaw command for FrankaPandaRobot (image angle -> world jaw axis)."""
        from simulation.grasp_executor import image_angle_to_gripper_yaw
        return image_angle_to_gripper_yaw(self.angle_rad)

    def __repr__(self):
        w_str = f"[{self.world_coords[0]:.3f}, {self.world_coords[1]:.3f}, {self.world_coords[2]:.3f}]" if self.world_coords is not None else "None"
        return f"Grasp(u={self.u:.1f}, v={self.v:.1f}, score={self.quality:.3f}, angle={self.angle_deg:.1f}deg, width={self.width_px:.1f}px, world={w_str})"
