import sys, os
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import numpy as np
import torch
from simulation.camera import OverheadCamera
from simulation.object_spawner import ObjectSpawner
from models.ggcnn import load_trained_model
import pybullet as p
import pybullet_data

p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

table_id = p.loadURDF("table/table.urdf", [0.5, 0.0, -0.65])

# Tightly frame the 30cm x 30cm workspace
cam = OverheadCamera(
    target_pos=(0.5, 0.0, 0.0),
    camera_pos=(0.5, 0.0, 0.52),
    img_size=(300, 300),
    fov=34.0
)

spawner = ObjectSpawner(
    workspace_bounds=((0.42, 0.58), (-0.10, 0.10)),
    table_height=0.0
)

spawner.spawn_random_objects(num_objects=3)
rgb, depth, seg = cam.capture()

# Test GGCNN
table_depth = 0.52
d = np.clip(depth, 0.0, table_depth)
d_norm = np.clip((d - d.mean()), -1.0, 1.0)
tensor = torch.from_numpy(d_norm).float().unsqueeze(0).unsqueeze(0).cuda()

model = load_trained_model('checkpoints/ggcnn_weights_cornell/ggcnn_epoch_23_cornell_statedict.pt', device='cuda')
with torch.no_grad():
    pos, cos, sin, w = model(tensor)

pos_np = pos.squeeze().cpu().numpy()
r, c = np.unravel_index(np.argmax(pos_np), pos_np.shape)
print(f"Max quality: {pos_np[r, c]:.3f} at pixel ({c}, {r})")
print(f"Is peak on an object (seg > 1)? {seg[r, c] > 1}")
world_pt = cam.deproject_pixel_to_world(c, r, depth[r, c])
print(f"World coords: {world_pt}")

p.disconnect()
