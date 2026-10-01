import numpy as np
import torch
import cv2
import time
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max

from .grasp_geometry import Grasp
from models.ggcnn import load_trained_model

class GraspDetector:
    """
    Real-time AI Grasp Detection Engine using GG-CNN.
    Analyzes RGB / Depth image streams, predicts pixel-wise grasp heatmaps,
    and returns ranked 3D antipodal grasp candidates.
    """
    def __init__(self,
                 checkpoint_path="checkpoints/ggcnn_weights_cornell/ggcnn_epoch_23_cornell_statedict.pt",
                 device=None,
                 input_channels=1,
                 min_quality_thresh=0.20,
                 gaussian_sigma=2.0,
                 table_depth=0.85):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.input_channels = input_channels
        self.min_quality_thresh = min_quality_thresh
        self.gaussian_sigma = gaussian_sigma
        self.table_depth = table_depth

        # Load model onto target device
        self.model = load_trained_model(
            checkpoint_path=checkpoint_path,
            input_channels=input_channels,
            device=self.device
        )
        print(f"GraspDetector initialized on: {self.device} (GPU: {torch.cuda.is_available()})")

    def preprocess(self, rgb, depth):
        """
        Preprocesses RGB and Depth for GG-CNN input.
        Normalizes depth with respect to workspace table surface using Morrison's formulation.
        """
        if self.input_channels == 1:
            # Depth input: clip floor at table plane, zero-center with mean, clip [-1, 1]
            d = np.clip(depth.copy(), 0.0, self.table_depth)
            d_norm = np.clip((d - d.mean()), -1.0, 1.0)
            tensor = torch.from_numpy(d_norm).float().unsqueeze(0).unsqueeze(0)
        elif self.input_channels == 3:
            # RGB input: normalize to [0, 1]
            rgb_norm = rgb.astype(np.float32) / 255.0
            tensor = torch.from_numpy(rgb_norm).permute(2, 0, 1).unsqueeze(0).float()
        else:
            # 4-channel RGB-D
            rgb_norm = rgb.astype(np.float32) / 255.0
            d = np.clip(depth.copy(), 0.0, self.table_depth)
            d_norm = np.clip((d - d.mean()), -1.0, 1.0)[:, :, None]
            rgbd = np.concatenate([rgb_norm, d_norm], axis=2)
            tensor = torch.from_numpy(rgbd).permute(2, 0, 1).unsqueeze(0).float()

        return tensor.to(self.device)

    def detect(self, rgb, depth, camera=None, top_k=5):
        """
        Runs CNN inference and extracts top-k ranked grasp candidates.
        Returns:
            grasps: list of Grasp objects (ranked by quality score)
            telemetry: dict containing inference_time_ms, fps, etc.
            heatmaps: dict containing smoothed 'q_map', 'ang_map', 'width_map'
        """
        inp_tensor = self.preprocess(rgb, depth)

        # Synchronize for precise GPU timing
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        t_start = time.perf_counter()

        with torch.no_grad():
            pos_out, cos_out, sin_out, width_out = self.model(inp_tensor)

        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t_start) * 1000.0
        fps = 1000.0 / max(latency_ms, 1e-4)

        # Convert outputs to numpy maps [H, W]
        q_raw = pos_out.squeeze().cpu().numpy()
        cos_raw = cos_out.squeeze().cpu().numpy()
        sin_raw = sin_out.squeeze().cpu().numpy()
        w_raw = width_out.squeeze().cpu().numpy()

        # Gaussian smoothing on quality heatmap
        q_smooth = gaussian_filter(q_raw, sigma=self.gaussian_sigma)
        # Compute angle theta in radians [-pi/2, pi/2]
        ang_map = 0.5 * np.arctan2(sin_raw, cos_raw)
        # Scale width (GG-CNN width output in pixels)
        width_map = np.clip(w_raw * 150.0, 20.0, 75.0)

        # Focus quality map on object regions (depth slightly above table)
        object_mask = (depth < (self.table_depth - 0.005))
        q_masked = q_smooth * object_mask.astype(np.float32)

        # Suppress image boundaries
        border = 15
        q_masked[:border, :] = 0.0
        q_masked[-border:, :] = 0.0
        q_masked[:, :border] = 0.0
        q_masked[:, -border:] = 0.0

        # Peak detection via local maxima with min separation
        peaks = peak_local_max(
            q_masked,
            min_distance=12,
            threshold_abs=self.min_quality_thresh,
            num_peaks=top_k
        )

        grasps = []
        for r, c in peaks:
            score = float(q_smooth[r, c])
            angle = float(ang_map[r, c])
            w_px = float(width_map[r, c])
            d_val = float(depth[r, c])

            # Deproject to 3D world coordinates if camera is provided
            world_pt = None
            if camera is not None:
                world_pt = camera.deproject_pixel_to_world(c, r, d_val)

            grasps.append(Grasp(
                center_px=(c, r),
                angle_rad=angle,
                width_px=w_px,
                quality=score,
                world_coords=world_pt
            ))

        # Sort descending by quality score
        grasps.sort(key=lambda g: g.quality, reverse=True)

        telemetry = {
            'inference_time_ms': latency_ms,
            'fps': fps,
            'device': str(self.device),
            'top_score': grasps[0].quality if len(grasps) > 0 else 0.0,
            'num_detected': len(grasps)
        }

        heatmaps = {
            'q_map': q_smooth,
            'ang_map': ang_map,
            'width_map': width_map
        }

        return grasps, telemetry, heatmaps
