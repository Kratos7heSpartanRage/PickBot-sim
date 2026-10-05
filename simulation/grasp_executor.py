"""
Physical grasp execution: turns a perceived Grasp into a 4-phase pick
(approach -> descend -> close -> lift [-> place]) on the simulated Panda,
with per-phase diagnostics (commanded vs. achieved pose, ground-truth object
pose, finger contact force) so failures are easy to localise.
"""
import numpy as np
import pybullet as p


def image_angle_to_gripper_yaw(angle_img):
    """
    Converts a grasp angle measured in the image (opening direction, radians,
    u-right / v-down) into the yaw command for FrankaPandaRobot.move_to_cartesian.

      * The camera's image v-axis points to world -Y, but we saw the CNN tracks
        angle_img, so we use it directly.
      * With orientation euler(pi, 0, yaw) the Panda jaw closing axis lies at
        world angle yaw + 90 deg (measured), so yaw = world_angle - 90 deg.
      * A parallel gripper is symmetric under 180 deg, so wrap into
        [-90, 90] deg to keep the wrist far from its joint limits.
    """
    yaw = angle_img - np.pi / 2.0
    return (yaw + np.pi / 2.0) % np.pi - np.pi / 2.0


class GraspExecutor:
    APPROACH_CLEARANCE = 0.12    # m above object top for the pre-grasp hover
    FINGER_DEPTH = 0.020         # m below object top to place the grasp frame
    TABLE_CLEARANCE = 0.012      # min grasp-frame height above table (tips ~7 mm lower)
    LIFT_HEIGHT = 0.25           # m above table
    SUCCESS_LIFT = 0.05          # object must rise this much to count as a pick

    def __init__(self, env, verbose=True, force_monitor=None):
        self.env = env
        self.robot = env.robot
        self.verbose = verbose
        self.force_monitor = force_monitor    # callable(phase, force, ee_pos) for GUI read-outs

    def _log(self, msg=""):
        if self.verbose:
            print(msg)

    # ------------------------------------------------------------------
    def _object_under(self, x, y):
        """Ground-truth object whose footprint contains (x, y), else the nearest one."""
        best, best_d = None, np.inf
        for data in self.env.spawner.object_data:
            try:
                lo, hi = p.getAABB(data["id"])
            except Exception:
                continue
            cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
            inside = lo[0] - 0.005 <= x <= hi[0] + 0.005 and lo[1] - 0.005 <= y <= hi[1] + 0.005
            d = 0.0 if inside else np.hypot(x - cx, y - cy)
            if d < best_d:
                best, best_d = (data, lo, hi), d
        return best, best_d

    def plan(self, grasp):
        """Computes grasp-frame targets from perception only (no ground truth)."""
        wx, wy, surface_z = grasp.world_coords
        table_z = self.env.table_z
        grasp_z = max(table_z + self.TABLE_CLEARANCE, surface_z - self.FINGER_DEPTH)
        return {
            "xy": np.array([wx, wy]),
            "surface_z": surface_z,
            "yaw": image_angle_to_gripper_yaw(grasp.angle_rad),
            "approach_z": max(surface_z, grasp_z) + self.APPROACH_CLEARANCE,
            "grasp_z": grasp_z,
            "lift_z": table_z + self.LIFT_HEIGHT,
        }

    def execute(self, grasp, place_in_tray=True, go_home=True):
        r = self.robot
        plan = self.plan(grasp)
        x, y = plan["xy"]
        yaw = plan["yaw"]
        obj_ids = set(self.env.spawner.spawned_ids)
        prev_verbose = r.verbose
        r.verbose = self.verbose

        gt, gt_dist = self._object_under(x, y)
        self._log(f"  Perceived  : xy=[{x:.3f}, {y:.3f}]  surface_z={plan['surface_z']:.3f}  "
                  f"img_angle={grasp.angle_deg:+.1f}°  → gripper yaw={np.degrees(yaw):+.1f}°")
        if gt is not None:
            data, lo, hi = gt
            self._log(f"  Ground trut: {data['name']}  center=[{(lo[0]+hi[0])/2:.3f}, {(lo[1]+hi[1])/2:.3f}]  "
                      f"z∈[{lo[2]:.3f}, {hi[2]:.3f}]  (xy offset {gt_dist*1000:.0f} mm"
                      f"{', INSIDE footprint' if gt_dist == 0 else ''})")
        z_before = {oid: p.getBasePositionAndOrientation(oid)[0][2] for oid in obj_ids}

        r.open_gripper(steps=20)

        self._log(f"  Phase 1 │ Approach → Z = {plan['approach_z']:.3f} m")
        r.move_to_cartesian([x, y, plan["approach_z"]], target_yaw=yaw, label="approach")

        self._log(f"  Phase 2 │ Descend  → Z = {plan['grasp_z']:.3f} m")
        res = r.move_to_cartesian([x, y, plan["grasp_z"]], target_yaw=yaw, speed=0.10, label="descend ")
        pre_contact = r.get_finger_contact_force(obj_ids)
        if pre_contact > 1.0:
            self._log(f"          │ ⚠ fingers already touching an object before closing ({pre_contact:.1f} N)")

        self._log("  Phase 3 │ Closing gripper (force-limited, monitoring contacts)...")
        monitor = (lambda f: self.force_monitor("GRIP", f, r.get_ee_pose()[0])) if self.force_monitor else None
        grip = r.close_gripper(force=60.0, steps=120, object_ids=obj_ids, monitor=monitor)
        opening = r.gripper_opening()
        self._log(f"          │ Peak grip force : {grip:.1f} N   finger opening: {opening*1000:.1f} mm")

        self._log(f"  Phase 4 │ Lift     → Z = {plan['lift_z']:.3f} m")
        r.move_to_cartesian([x, y, plan["lift_z"]], target_yaw=yaw, speed=0.15, label="lift    ")
        r.step(30)
        lift_force = r.get_finger_contact_force(obj_ids)
        if self.force_monitor:
            self.force_monitor("LIFT", lift_force, r.get_ee_pose()[0])
        self._log(f"          │ Holding force after lift : {lift_force:.1f} N")

        lifted = None
        for data in self.env.spawner.object_data:
            oid = data["id"]
            z_now = p.getBasePositionAndOrientation(oid)[0][2]
            if z_now - z_before.get(oid, z_now) > self.SUCCESS_LIFT and lift_force > 0.5:
                lifted = data
                break

        result = {
            "success": lifted is not None,
            "object": lifted["name"] if lifted else None,
            "grip_force": grip,
            "lift_force": lift_force,
            "descend_error_m": res["tracking_error_m"],
            "plan": plan,
        }

        if lifted is not None:
            self._log(f"  Result  │ ✓ SUCCESS — {lifted['name']} lifted  (grip {grip:.1f} N, hold {lift_force:.1f} N)")
            if place_in_tray and self.env.tray_id is not None:
                tray_xy = p.getBasePositionAndOrientation(self.env.tray_id)[0][:2]
                self._log("          │ Placing in collection tray...")
                r.move_to_cartesian([tray_xy[0], tray_xy[1], plan["lift_z"]], target_yaw=0.0, label="to tray ")
                r.move_to_cartesian([tray_xy[0], tray_xy[1], self.env.table_z + 0.15], target_yaw=0.0, speed=0.1)
                r.open_gripper(steps=60)
                r.move_to_cartesian([tray_xy[0], tray_xy[1], plan["lift_z"]], target_yaw=0.0)
        else:
            if grip < 0.5:
                why = "fingers closed on nothing (grasp point/height missed the object)"
            elif opening < 0.004:
                why = "object squeezed out of the fingers"
            else:
                why = "contact made but object slipped during lift"
            self._log(f"  Result  │ ✗ MISS — {why}")
            r.open_gripper(steps=30)

        if go_home:
            r.go_home()
        r.verbose = prev_verbose
        return result
