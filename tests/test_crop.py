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

print("Segmentation IDs present in overhead image:")
unique, counts = np.unique(seg, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  ID {u}: {c} pixels ({c/(300*300)*100:.1f}%)")

print("Robot body ID:", env.robot.robot_id)
print("Table body ID:", env.table_id)
print("Spawned object IDs:", env.spawner.spawned_ids)

env.close()
