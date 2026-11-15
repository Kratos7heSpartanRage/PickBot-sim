import sys, os
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import torch

ckpt = torch.load('checkpoints/ggcnn_weights_cornell/ggcnn_epoch_23_cornell_statedict.pt', map_location='cpu')
print('Number of layers:', len(ckpt))
for k, v in ckpt.items():
    print(f'{k}: {v.shape}')
