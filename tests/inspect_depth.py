import sys, os
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import numpy as np
from simulation.environment import GraspingEnvironment

env = GraspingEnvironment(gui=False)
env.reset(num_objects=3)
rgb, depth, seg = env.get_observation()

print("Depth stats:")
print(f"  Min depth: {np.min(depth):.3f}m")
print(f"  Max depth: {np.max(depth):.3f}m")
print(f"  Median depth (table): {np.median(depth):.3f}m")

# Object depth vs table depth
obj_mask = seg > 1
if np.any(obj_mask):
    print(f"  Object depth mean: {np.mean(depth[obj_mask]):.3f}m")
    print(f"  Table depth mean: {np.mean(depth[seg == 1]):.3f}m")
    print(f"  Object height above table: {np.mean(depth[seg == 1]) - np.mean(depth[obj_mask]):.3f}m")

env.close()
