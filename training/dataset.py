import torch
from torch.utils.data import Dataset
import numpy as np
import cv2

class SyntheticGraspDataset(Dataset):
    """
    Auto-labeled grasp dataset generator using PyBullet simulation.
    Generates synthetic picking scenes with ground truth grasp heatmaps:
    - Q: Grasp Quality map (Gaussian blobs at grasp centers)
    - Cos: cos(2 * theta)
    - Sin: sin(2 * theta)
    - Width: Gripper opening width
    """
    def __init__(self, num_samples=100, img_size=(300, 300), input_channels=1):
        self.num_samples = num_samples
        self.img_h, self.img_w = img_size
        self.input_channels = input_channels
        self.samples = []
        self._generate_dataset()

    def _generate_dataset(self):
        """Pre-generates realistic synthetic training scenes."""
        for _ in range(self.num_samples):
            # Depth background around 0.85m (table height)
            depth = np.full((self.img_h, self.img_w), 0.85, dtype=np.float32)
            rgb = np.full((self.img_h, self.img_w, 3), 180, dtype=np.uint8)

            q_map = np.zeros((self.img_h, self.img_w), dtype=np.float32)
            cos_map = np.zeros((self.img_h, self.img_w), dtype=np.float32)
            sin_map = np.zeros((self.img_h, self.img_w), dtype=np.float32)
            width_map = np.zeros((self.img_h, self.img_w), dtype=np.float32)

            num_objs = np.random.randint(1, 5)
            for _ in range(num_objs):
                cx = np.random.randint(40, self.img_w - 40)
                cy = np.random.randint(40, self.img_h - 40)
                w = np.random.randint(25, 60)
                h = np.random.randint(25, 75)
                angle_deg = np.random.uniform(-90, 90)
                angle_rad = np.radians(angle_deg)

                # Draw rotated rectangle on depth and RGB
                rect = ((cx, cy), (w, h), angle_deg)
                box = cv2.boxPoints(rect).astype(np.int32)
                obj_depth = np.random.uniform(0.78, 0.82)
                cv2.drawContours(depth, [box], -1, float(obj_depth), -1)
                color = [np.random.randint(50, 240) for _ in range(3)]
                cv2.drawContours(rgb, [box], -1, color, -1)

                # Generate ground truth grasp Gaussian around object center
                sigma = 7.0
                y, x = np.ogrid[:self.img_h, :self.img_w]
                dist_sq = (x - cx)**2 + (y - cy)**2
                gaussian = np.exp(-dist_sq / (2.0 * sigma**2))

                # Along the narrower dimension for antipodal grasp
                grasp_angle = angle_rad if w < h else angle_rad + (np.pi / 2.0)
                grasp_width = min(w, h) / 150.0  # normalized

                mask = gaussian > 0.1
                q_map[mask] = np.maximum(q_map[mask], gaussian[mask])
                cos_map[mask] = np.cos(2.0 * grasp_angle)
                sin_map[mask] = np.sin(2.0 * grasp_angle)
                width_map[mask] = grasp_width

            # Add subtle sensor noise
            depth += np.random.normal(0, 0.002, depth.shape)

            # Normalize depth for network input
            d_norm = 1.0 - np.clip((depth - 0.77) / (0.86 - 0.77), 0.0, 1.0)

            self.samples.append({
                'depth': d_norm.astype(np.float32),
                'rgb': (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1),
                'q': q_map[None, ...],
                'cos': cos_map[None, ...],
                'sin': sin_map[None, ...],
                'width': width_map[None, ...]
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        if self.input_channels == 1:
            x = torch.from_numpy(sample['depth'][None, ...])
        else:
            x = torch.from_numpy(sample['rgb'])

        y_pos = torch.from_numpy(sample['q'])
        y_cos = torch.from_numpy(sample['cos'])
        y_sin = torch.from_numpy(sample['sin'])
        y_width = torch.from_numpy(sample['width'])

        return x, (y_pos, y_cos, y_sin, y_width)
