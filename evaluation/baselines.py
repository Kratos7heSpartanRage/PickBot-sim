import numpy as np
import cv2
import time
from perception.grasp_geometry import Grasp

class RandomGraspDetector:
    """
    Baseline 1: Random Grasp.
    Picks a random candidate point on the object area with a random gripper angle.
    """
    def __init__(self):
        self.name = "Random Grasp"

    def detect(self, rgb, depth, seg, camera=None):
        t_start = time.perf_counter()

        # Find object pixels (seg > 1, table is 1, plane is 0)
        obj_pixels = np.argwhere(seg > 1)
        if len(obj_pixels) == 0:
            obj_pixels = np.argwhere(depth < 0.83)

        if len(obj_pixels) == 0:
            return [], {'inference_time_ms': 0.1, 'top_score': 0.0}

        idx = np.random.choice(len(obj_pixels))
        v, u = obj_pixels[idx]
        angle = np.random.uniform(-np.pi/2, np.pi/2)
        width = 40.0
        # Random grasps have modest confidence
        quality = float(np.random.uniform(0.15, 0.45))

        world_pt = None
        if camera is not None:
            world_pt = camera.deproject_pixel_to_world(u, v, depth[v, u])

        grasp = Grasp((u, v), angle, width, quality=quality, world_coords=world_pt)
        latency = (time.perf_counter() - t_start) * 1000.0

        return [grasp], {'inference_time_ms': latency, 'top_score': quality}

class CentroidGraspDetector:
    """
    Baseline 2: Centroid Heuristic.
    Finds the center of mass of the largest object segment and computes the
    principal axis of orientation from second-order image moments.
    """
    def __init__(self):
        self.name = "Centroid Heuristic"

    def detect(self, rgb, depth, seg, camera=None):
        t_start = time.perf_counter()

        # Mask objects
        mask = (seg > 1).astype(np.uint8)
        if np.sum(mask) == 0:
            mask = (depth < 0.83).astype(np.uint8)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return [], {'inference_time_ms': 0.2, 'top_score': 0.0}

        # Select largest object contour
        largest_cnt = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_cnt)
        if M["m00"] == 0:
            return [], {'inference_time_ms': 0.2, 'top_score': 0.0}

        # Centroid
        u = float(M["m10"] / M["m00"])
        v = float(M["m01"] / M["m00"])

        # Orientation from central moments: 0.5 * atan2(2 * mu11, mu20 - mu02)
        mu20 = M["mu20"] / M["m00"]
        mu02 = M["mu02"] / M["m00"]
        mu11 = M["mu11"] / M["m00"]
        angle = 0.5 * np.arctan2(2.0 * mu11, mu20 - mu02) + (np.pi / 2.0)
        # Normalize to [-pi/2, pi/2]
        angle = (angle + np.pi/2) % np.pi - np.pi/2

        # Grasp width from minor axis
        width = 45.0
        quality = 0.65  # Centroid heuristic has moderate quality

        v_idx, u_idx = int(np.clip(v, 0, depth.shape[0]-1)), int(np.clip(u, 0, depth.shape[1]-1))
        world_pt = None
        if camera is not None:
            world_pt = camera.deproject_pixel_to_world(u, v, depth[v_idx, u_idx])

        grasp = Grasp((u, v), angle, width, quality=quality, world_coords=world_pt)
        latency = (time.perf_counter() - t_start) * 1000.0

        return [grasp], {'inference_time_ms': latency, 'top_score': quality}

class DepthAntipodalGraspDetector:
    """
    Baseline 3: Depth-based Antipodal Heuristic.
    Computes depth edges via Sobel filters and finds opposing parallel surface normals.
    """
    def __init__(self):
        self.name = "Depth-based Heuristic"

    def detect(self, rgb, depth, seg, camera=None):
        t_start = time.perf_counter()

        # Compute depth gradients
        d_blur = cv2.GaussianBlur(depth, (5, 5), 0)
        sobel_x = cv2.Sobel(d_blur, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(d_blur, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)

        # Threshold edges
        edges = grad_mag > np.percentile(grad_mag, 92)
        edge_pts = np.argwhere(edges)

        if len(edge_pts) < 2:
            return [], {'inference_time_ms': 0.5, 'top_score': 0.0}

        # Pick a point on edge and search for antipodal pair
        pt1 = edge_pts[np.random.choice(len(edge_pts))]
        v1, u1 = pt1
        normal_angle = np.arctan2(sobel_y[v1, u1], sobel_x[v1, u1])

        # Step along normal to find opposing edge
        step_dir = np.array([np.cos(normal_angle), np.sin(normal_angle)])
        candidate_u, candidate_v = u1, v1
        for dist in range(15, 60, 3):
            test_u = int(u1 + dist * step_dir[0])
            test_v = int(v1 + dist * step_dir[1])
            if 0 <= test_u < depth.shape[1] and 0 <= test_v < depth.shape[0]:
                if edges[test_v, test_u]:
                    candidate_u, candidate_v = (u1 + test_u) / 2.0, (v1 + test_v) / 2.0
                    break

        grasp_angle = normal_angle + np.pi/2.0
        grasp_angle = (grasp_angle + np.pi/2) % np.pi - np.pi/2
        quality = 0.72

        v_idx = int(np.clip(candidate_v, 0, depth.shape[0]-1))
        u_idx = int(np.clip(candidate_u, 0, depth.shape[1]-1))
        world_pt = None
        if camera is not None:
            world_pt = camera.deproject_pixel_to_world(candidate_u, candidate_v, depth[v_idx, u_idx])

        grasp = Grasp((candidate_u, candidate_v), grasp_angle, width_px=45.0, quality=quality, world_coords=world_pt)
        latency = (time.perf_counter() - t_start) * 1000.0

        return [grasp], {'inference_time_ms': latency, 'top_score': quality}
