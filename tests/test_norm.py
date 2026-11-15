import sys, os
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import numpy as np
import torch
from simulation.environment import GraspingEnvironment
from models.ggcnn import load_trained_model

env = GraspingEnvironment(gui=False)
env.reset(num_objects=3)
rgb, depth, seg = env.get_observation()

# Clip depth to table plane (so anything below table is table depth)
table_depth = 0.85
d = np.clip(depth.copy(), 0.0, table_depth)
d_norm = np.clip((d - d.mean()), -1.0, 1.0)
tensor = torch.from_numpy(d_norm).float().unsqueeze(0).unsqueeze(0).cuda()

model = load_trained_model('checkpoints/ggcnn_weights_cornell/ggcnn_epoch_23_cornell_statedict.pt', device='cuda')
with torch.no_grad():
    pos, cos, sin, w = model(tensor)

pos_np = pos.squeeze().cpu().numpy()
print('pos min/max/mean:', pos_np.min(), pos_np.max(), pos_np.mean())
peak_r, peak_c = np.unravel_index(np.argmax(pos_np), pos_np.shape)
print(f'Peak pos location: ({peak_r}, {peak_c}), value: {pos_np[peak_r, peak_c]:.4f}')
print('Is peak on an object (seg > 1)?', seg[peak_r, peak_c] > 1)
env.close()
