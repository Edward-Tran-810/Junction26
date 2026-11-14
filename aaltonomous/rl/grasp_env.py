"""Aalto-Grasp-G129-Dex3-RL: vectorised grasp-and-lift training env (left arm), headless, no DDS, no cameras.

Same robot (upstream G1 29-DoF + Dex3 base-fix preset, sim ARM_HOME), counter and object configs as the Aalto task
(imported from sim/aalto_task/scene_cfg.py, values not copied), same physics settings (dt 5 ms, solver / CCD /
contact settings of env_cfg.py). Policy step = 30 Hz with the DDS sim's zero-order hold: the eval task steps
physics in 10 ms env steps and a 30 Hz target lands on the next env step, i.e. holds of 8 / 6 / 6 physics steps.
Differences from the eval task (measured in round 0, see STATE.md): physics on GPU (eval sim: CPU), no bins / mat /
other objects, the arm STARTS at the pregrasp (eval: moves there from home).

Per env: 3 objects (one per shape) from scene_cfg.object_cfgs(); the episode's shape is placed on the counter, the
other two are parked on the floor behind the robot.

Episode (scripted.py library entry): the scripted pregrasp -> descend -> close -> lift -> hold 1 s -> shake -> hold,
planned by the oracle planner for the ESTIMATED pose; the true object = estimate + placement noise (+ wrong shape
with prob p). Target each tick = script + bounded residual, projected so that the change per tick never exceeds the
velocity caps and the target stays inside the safety filter's joint range (0 clips by construction).

Success: object centre >= 10 cm above its rest height on every tick of the hold phase AND at the end of the
episode (after the shake), and no counter keep-out / workspace violation by the left wrist/palm (measured link
positions, 1 cm margin: a proxy of the filter's FK check on the commanded target).
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING

import numpy as np
import torch

from . import _paths  # noqa: F401  (must precede scene_cfg)
import isaaclab.sim as sim_utils
from isaaclab_physx.physics import PhysxCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from aaltonomous.interface.safety import JointSpec, SafetyConfig
from aaltonomous.sim.aalto_task import scene_cfg as SC
from aaltonomous.sim.aalto_task.layout import ARM_HOME, PELVIS_W

from . import scripted as S

LIB_DEFAULT = str(SC.REPO / "aaltonomous/rl/data/lib_left.npz")
SHAPES = S.SHAPES
HOLD_PATTERN = (8, 6, 6)          # physics steps per 30 Hz tick (sum 20 x 5 ms = 0.1 s = 3 ticks)
ARM_VEL_CAP, HAND_VEL_CAP = 1.0, 3.0
CAP_FRAC = 0.97                   # use 97 % of the cap (float + filter rounding)
LIFT_M = 0.10
# Left grasp point in the palm frame (= ORACLE_OVERRIDES grasp.point_in_palm) and Dex3 fingertips in their links.
GRASP_IN_PALM = (0.099, -0.084, 0.015)
TIP_LINKS = ("left_hand_index_1_link", "left_hand_middle_1_link", "left_hand_thumb_2_link")
TIP_OFFSETS = ((0.052, 0.0, 0.0), (0.052, 0.0, 0.0), (0.0, -0.052, 0.0))
# Action spaces: indices into the 14-vector [left arm 7, left hand 7 (left DDS order)].
ACTION_DIMS = {"hand": list(range(7, 14)), "hand_wrist": [4, 5, 6] + list(range(7, 14)), "full": list(range(14))}


def _park_xy(i):
    return (-0.60, -0.3 + 0.3 * i)


@configclass
class GraspSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = MISSING
    counter: AssetBaseCfg = MISSING
    floor: AssetBaseCfg = MISSING
    obj_cube: RigidObjectCfg = MISSING
    obj_cylinder: RigidObjectCfg = MISSING
    obj_triangle: RigidObjectCfg = MISSING


def make_scene_cfg(num_envs, env_spacing=2.5):
    fixed = SC._fixed_cfgs()
    objs = SC.object_cfgs()
    out = {}
    for i, shape in enumerate(SHAPES):
        c = objs[f"obj_red_{shape}_0"].replace(prim_path=f"{SC.ENV}/obj_{shape}")
        x, y = _park_xy(i)
        c.init_state.pos = (x, y, PELVIS_W[2] * 0 + 0.03)
        out[f"obj_{shape}"] = c
    return GraspSceneCfg(num_envs=num_envs, env_spacing=env_spacing, replicate_physics=True,
                         robot=fixed["robot"], counter=fixed["counter"], floor=fixed["floor"], **out)


@configclass
class GraspEnvCfg(DirectRLEnvCfg):
    # ---- physics / timing (eval task values, env_cfg.py)
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(dt=0.005, render_interval=1000,
                                                       physics=PhysxCfg(enable_external_forces_every_iteration=False))
    decimation: int = 7                    # nominal; step() uses HOLD_PATTERN
    episode_length_s: float = 10.0         # upper bound; the library entry's length ends the episode
    scene: GraspSceneCfg = None
    num_envs: int = 256
    # ---- spaces (filled in __post_init__)
    action_space: int = 14
    observation_space: int = 0
    state_space: int = 0
    # ---- library / episode
    lib_path: str = LIB_DEFAULT
    action_space_name: str = "full"        # hand | hand_wrist | full
    residual_arm: float = 0.12             # rad, max |offset| per arm joint
    residual_hand: float = 0.5             # rad, max |offset| per finger joint
    # residual = EMA of the actions (low-pass): per-tick white exploration noise vibrates the kp-100 fingers and
    # drops the scripted grasp (scripted + N(0, 0.05) actions: cylinder 68/511 vs ~88 % without noise)
    action_ema: float = 0.2
    # ---- object estimate (actor) vs truth
    use_estimate: bool = True              # False = proprioception-only actor
    est_pos_noise: float = 0.005           # m, uniform +- (truth = estimate + noise)
    est_yaw_noise_deg: float = 10.0
    est_wrong_shape_p: float = 0.05
    joint_noise: float = 0.003             # rad on arm_q / hand_q readings; arm_dq gets 10x
    # ---- physics DR (sampled once per env at startup; eval overrides with fixed values)
    friction_mode: str = "narrow"          # narrow: object x [0.75, 1.1] (train.yaml) | wide: obj + fingers 0.3-1.2
    friction_fixed: float | None = None    # eval: object (and fingers if fingers_match) at exactly this static friction
    # True: every robot shape gets the object's friction, so object-finger contact = the sampled value. False: the
    # robot keeps the USD materials = the eval sim (Dex3 links mu 5.0 except the left index 0.5; contact = max).
    fingers_match: bool = True
    mass_scale: tuple = (0.7, 1.5)
    finger_model: str = "default"          # default | random
    finger_kp_scale: tuple = (0.5, 2.0)
    finger_kd_scale: tuple = (0.5, 2.0)
    finger_armature_scale: tuple = (0.2, 2.0)
    finger_friction: tuple = (0.0, 0.05)   # joint (Coulomb) friction, Nm
    # ---- reward
    w_success: float = 10.0
    w_lift: float = 0.05                   # per tick, x clamp(rise / 10 cm) after the close phase starts
    w_action_rate: float = 0.02
    w_counter: float = 0.2                 # per tick a fingertip is within 3 mm of the counter (or below)
    w_violation: float = 1.0

    def __post_init__(self):
        self.scene = make_scene_cfg(self.num_envs)
        p = self.sim.physics
        p.bounce_threshold_velocity = 0.01
        p.friction_correlation_distance = 0.003
        p.enable_ccd = True
        p.min_position_iteration_count = 16
        p.min_velocity_iteration_count = 4
        n = max(self.num_envs, 64)
        p.gpu_found_lost_aggregate_pairs_capacity = max(1024 * 1024 * 4, n * 4096)
        p.gpu_total_aggregate_pairs_capacity = max(32 * 1024, n * 256)
        p.gpu_max_rigid_contact_count = max(2 ** 23, n * 2048)
        p.gpu_max_rigid_patch_count = max(2 ** 18, n * 512)
        p.gpu_found_lost_pairs_capacity = max(2 ** 21, n * 1024)
        self.action_space = len(ACTION_DIMS[self.action_space_name])
        self.observation_space = 7 + 7 + 7 + 14 + 8 + self.action_space + 11
        self.state_space = self.observation_space + 3 + 4 + 6 + 1 + 9 + 3 + 6


class GraspEnv(DirectRLEnv):
    cfg: GraspEnvCfg

    def __init__(self, cfg: GraspEnvCfg, render_mode=None, **kw):
        super().__init__(cfg, render_mode, **kw)
        dev = self.device
        N = self.num_envs
        self.robot = self.scene["robot"]
        self.objs = [self.scene[f"obj_{s}"] for s in SHAPES]
        self.spec = JointSpec()
        names = self.robot.joint_names
        self.arm_ids = torch.tensor([names.index(n) for n in self.spec.arm_names], device=dev)
        self.hand_ids = torch.tensor([names.index(n) for n in self.spec.hand_names], device=dev)
        self.left_ids = torch.cat([self.arm_ids[:7], self.hand_ids[:7]])
        self.ctrl_ids = torch.cat([self.arm_ids, self.hand_ids])            # 28 commanded joints
        # filter joint range (URDF minus margin) for the 28 commanded joints
        sc = SafetyConfig()
        lo = np.r_[self.spec.arm_lower + sc.arm_pos_margin, self.spec.hand_lower + sc.hand_pos_margin]
        hi = np.r_[self.spec.arm_upper - sc.arm_pos_margin, self.spec.hand_upper - sc.hand_pos_margin]
        self.lim_lo = torch.tensor(lo, dtype=torch.float32, device=dev)
        self.lim_hi = torch.tensor(hi, dtype=torch.float32, device=dev)
        dt = 1.0 / 30.0
        self.step_cap = torch.tensor([ARM_VEL_CAP * dt * CAP_FRAC] * 14 + [HAND_VEL_CAP * dt * CAP_FRAC] * 14,
                                     device=dev)
        # bodies
        bn = self.robot.body_names
        self.palm_id = bn.index("left_hand_palm_link")
        self.wrist_id = bn.index("left_wrist_yaw_link")
        self.tip_ids = torch.tensor([bn.index(n) for n in TIP_LINKS], device=dev)
        self.tip_off = torch.tensor(TIP_OFFSETS, dtype=torch.float32, device=dev)
        self.grasp_off = torch.tensor(GRASP_IN_PALM, dtype=torch.float32, device=dev)
        self.pelvis_w = torch.tensor(PELVIS_W, dtype=torch.float32, device=dev)
        self.counter_z = float(SC.SC.counter_z)
        self.obj_h = {s: float(SC.SC.object_height(s)) for s in SHAPES}
        # library
        L = np.load(cfg.lib_path)
        t = lambda k, dt_=torch.float32: torch.tensor(L[k], dtype=dt_, device=dev)
        self.lib = {k: t(k) for k in ("x", "y", "yaw_pelvis", "start_arm", "start_hand", "arm", "hand")}
        self.lib["shape"] = t("shape", torch.long)
        self.lib["length"] = t("length", torch.long)
        self.lib["starts"] = t("starts", torch.long)
        self.n_lib, self.T_lib = self.lib["arm"].shape[:2]
        # action
        self.act_dims = torch.tensor(ACTION_DIMS[cfg.action_space_name], device=dev)
        scale = torch.tensor([cfg.residual_arm] * 7 + [cfg.residual_hand] * 7, device=dev)
        self.act_scale = scale[self.act_dims]
        # per-env state
        z = lambda *s, dt_=torch.float32: torch.zeros(*s, dtype=dt_, device=dev)
        self.entry = z(N, dt_=torch.long)
        self.true_shape = z(N, dt_=torch.long)
        self.t = z(N, dt_=torch.long)
        self.target = z(N, 28)
        self.actions = z(N, cfg.action_space)
        self.prev_actions = z(N, cfg.action_space)
        self.res_a = z(N, cfg.action_space)            # smoothed residual in action units [-1, 1]
        self.prev_res_a = z(N, cfg.action_space)
        self.rest_z = z(N)
        self.held_hold = torch.ones(N, dtype=torch.bool, device=dev)
        self.violated = torch.zeros(N, dtype=torch.bool, device=dev)
        self.counter_ticks = z(N)
        self.min_obj_dz = z(N)
        self.success = torch.zeros(N, dtype=torch.bool, device=dev)
        self.lifted_any = torch.zeros(N, dtype=torch.bool, device=dev)
        self.ep_stats = []                  # finished episodes (eval mode collects these)
        self.collect_stats = False
        self._tick = 0
        # default joint pose (legs/waist from the preset, arms ARM_HOME, hands open)
        self.default_q = self.robot.data.default_joint_pos.torch.clone()
        self._setup_physics_dr()

    # ------------------------------------------------------------------ physics DR (startup)
    def _setup_physics_dr(self):
        cfg, N, dev = self.cfg, self.num_envs, self.device
        g = torch.Generator(device="cpu").manual_seed(int(cfg.seed or 0) + 12345)
        U = lambda lo, hi, n=N: lo + (hi - lo) * torch.rand(n, generator=g)
        # friction: object static mu; dynamic = mu * 1.0/1.2 (nominal ratio)
        if cfg.friction_fixed is not None:
            mu_o = torch.full((N,), float(cfg.friction_fixed))
        elif cfg.friction_mode == "wide":
            mu_o = U(0.3, 1.2)
        else:
            mu_o = SC.OBJ_STATIC_FRICTION * U(0.75, 1.1)
        mu_f = mu_o.clone() if cfg.fingers_match else None    # None: fingers keep the asset's own materials
        self.mu_obj = mu_o.to(dev)
        ids = torch.arange(N, dtype=torch.int32)
        import warp as wp
        for o in self.objs:
            m = wp.to_torch(o.root_view.get_material_properties()).clone()
            m[:, :, 0] = mu_o[:, None]
            m[:, :, 1] = mu_o[:, None] * (SC.OBJ_DYNAMIC_FRICTION / SC.OBJ_STATIC_FRICTION)
            o.root_view.set_material_properties(wp.from_torch(m.contiguous(), dtype=wp.float32),
                                                wp.from_torch(ids, dtype=wp.int32))
        rm = wp.to_torch(self.robot.root_view.get_material_properties()).clone()
        self.robot_mat_nominal = rm[0].clone()
        if mu_f is not None:
            rm[:, :, 0] = mu_f[:, None]
            rm[:, :, 1] = mu_f[:, None] * (SC.OBJ_DYNAMIC_FRICTION / SC.OBJ_STATIC_FRICTION)
            self.robot.root_view.set_material_properties(wp.from_torch(rm.contiguous(), dtype=wp.float32),
                                                         wp.from_torch(ids, dtype=wp.int32))
        self.mu_fing = (mu_f if mu_f is not None else torch.full((N,), float(rm[0, :, 0].max()))).to(dev)
        # mass
        lo, hi = cfg.mass_scale
        self.mass_scale = U(lo, hi).to(dev)
        for o in self.objs:
            m = o.data.body_mass.torch.clone()               # (N, 1)
            m[:] = SC.OBJ_MASS * self.mass_scale[:, None]
            o.set_masses_index(masses=m)
        # finger model (left + right hand joints)
        hid = self.hand_ids
        kp0 = self.robot.data.joint_stiffness.torch[:, hid].clone()
        kd0 = self.robot.data.joint_damping.torch[:, hid].clone()
        ar0 = self.robot.data.joint_armature.torch[:, hid].clone()
        self.finger_scales = torch.ones(N, 4, device=dev)
        if cfg.finger_model == "random":
            s = torch.stack([U(*cfg.finger_kp_scale), U(*cfg.finger_kd_scale), U(*cfg.finger_armature_scale),
                             U(*cfg.finger_friction)], 1).to(dev)
            self.finger_scales = s
            self.robot.write_joint_stiffness_to_sim_index(stiffness=kp0 * s[:, :1], joint_ids=hid)
            self.robot.write_joint_damping_to_sim_index(damping=kd0 * s[:, 1:2], joint_ids=hid)
            self.robot.write_joint_armature_to_sim_index(armature=ar0 * s[:, 2:3], joint_ids=hid)
            self.robot.write_joint_friction_coefficient_to_sim_index(
                joint_friction_coeff=s[:, 3:4].expand(-1, len(hid)).contiguous(), joint_ids=hid)
        self.finger_nominal = (kp0[0].tolist(), kd0[0].tolist(), ar0[0].tolist())

    # ------------------------------------------------------------------ stepping
    def step(self, action):
        self.cfg.decimation = HOLD_PATTERN[self._tick % 3]
        self._tick += 1
        return super().step(action)

    def _script(self, t):
        e = self.entry
        tt = torch.clamp(t, max=self.T_lib - 1)
        return torch.cat([self.lib["arm"][e, tt], self.lib["hand"][e, tt]], 1)    # (N, 28)

    def _pre_physics_step(self, actions):
        self.prev_actions[:] = self.actions
        self.actions[:] = torch.clamp(actions, -1.0, 1.0)
        res = torch.zeros(self.num_envs, 14, device=self.device)
        self.prev_res_a[:] = self.res_a
        self.res_a.mul_(1.0 - self.cfg.action_ema).add_(self.cfg.action_ema * self.actions)
        res[:, self.act_dims] = self.res_a * self.act_scale
        desired = self._script(self.t)
        desired[:, 0:7] += res[:, 0:7]
        desired[:, 14:21] += res[:, 7:14]
        lo = self.target - self.step_cap
        hi = self.target + self.step_cap
        tgt = torch.maximum(torch.minimum(desired, hi), lo)
        self.target[:] = torch.maximum(torch.minimum(tgt, self.lim_hi), self.lim_lo)
        self.t += 1

    def _apply_action(self):
        self.robot.actuators.target_command.set_position_index(value=self.target, joint_ids=self.ctrl_ids)

    # ------------------------------------------------------------------ state helpers
    def _obj_state(self):
        """True object (active shape) pos (pelvis frame), quat xyzw, lin vel, ang vel."""
        N = self.num_envs
        pos = torch.zeros(N, 3, device=self.device)
        quat = torch.zeros(N, 4, device=self.device)
        vel = torch.zeros(N, 6, device=self.device)
        for i, o in enumerate(self.objs):
            m = (self.true_shape == i)[:, None]
            pos = torch.where(m, o.data.root_pos_w.torch - self.scene.env_origins - self.pelvis_w, pos)
            quat = torch.where(m, o.data.root_quat_w.torch, quat)
            vel = torch.where(m, o.data.root_vel_w.torch, vel)
        return pos, quat, vel

    def _hand_geom(self):
        org = self.scene.env_origins[:, None] + self.pelvis_w
        bp = self.robot.data.body_pos_w.torch
        bq = self.robot.data.body_quat_w.torch
        from isaaclab.utils.math import quat_apply
        palm_p, palm_q = bp[:, self.palm_id] - org[:, 0], bq[:, self.palm_id]
        grasp = palm_p + quat_apply(palm_q, self.grasp_off.expand(self.num_envs, 3))
        tq = bq[:, self.tip_ids]
        tips = bp[:, self.tip_ids] - org + quat_apply(tq.reshape(-1, 4), self.tip_off.repeat(self.num_envs, 1)
                                                      ).reshape(self.num_envs, 3, 3)
        wrist = bp[:, self.wrist_id] - org[:, 0]
        return grasp, tips, palm_p, wrist

    def _phase(self):
        st = self.lib["starts"][self.entry]                 # (N, 6)
        ph = (self.t[:, None] >= st).sum(1) - 1             # 0..5
        return ph.clamp(0, 5)

    # ------------------------------------------------------------------ observations
    def _get_observations(self):
        N, dev, c = self.num_envs, self.device, self.cfg
        q = self.robot.data.joint_pos.torch
        dq = self.robot.data.joint_vel.torch
        arm_q, arm_dq, hand_q = q[:, self.arm_ids[:7]], dq[:, self.arm_ids[:7]], q[:, self.hand_ids[:7]]
        n = c.joint_noise
        if n > 0:
            arm_q = arm_q + n * torch.randn_like(arm_q)
            arm_dq = arm_dq + 10 * n * torch.randn_like(arm_dq)
            hand_q = hand_q + n * torch.randn_like(hand_q)
        scr = self._script(self.t)
        scr14 = torch.cat([scr[:, 0:7], scr[:, 14:21]], 1)
        ph = self._phase()
        st = self.lib["starts"][self.entry]
        L = self.lib["length"][self.entry]
        ph_oh = torch.nn.functional.one_hot(ph, 6).float()
        prog = torch.stack([(self.t / L.clamp(min=1)).float(),
                            ((self.t - st.gather(1, ph[:, None])[:, 0]).float() / 30.0)], 1)
        e = self.entry
        yaw = self.lib["yaw_pelvis"][e]
        est = torch.cat([torch.nn.functional.one_hot(self.lib["shape"][e], 3).float(),
                         torch.stack([(self.lib["x"][e] - 0.30) * 10, (self.lib["y"][e] - 0.08) * 10], 1),
                         torch.stack([torch.sin(yaw), torch.cos(yaw), torch.sin(3 * yaw), torch.cos(3 * yaw),
                                      torch.sin(4 * yaw), torch.cos(4 * yaw)], 1)], 1)
        if not c.use_estimate:
            est = torch.zeros_like(est)
        actor = torch.cat([arm_q, arm_dq, hand_q, scr14, ph_oh, prog, self.res_a, est], 1)
        # critic: noise-free proprio + true state
        pos, quat, vel = self._obj_state()
        grasp, tips, palm, wrist = self._hand_geom()
        rise = pos[:, 2] - self.rest_z
        critic_extra = torch.cat([pos - grasp, quat, vel, rise[:, None], (tips - pos[:, None]).reshape(N, 9),
                                  torch.nn.functional.one_hot(self.true_shape, 3).float(),
                                  self.mu_obj[:, None], self.mu_fing[:, None], self.mass_scale[:, None],
                                  self.finger_scales[:, :3]], 1)
        critic_base = torch.cat([q[:, self.arm_ids[:7]], dq[:, self.arm_ids[:7]], q[:, self.hand_ids[:7]],
                                 actor[:, 21:]], 1)
        return {"policy": actor, "critic": torch.cat([critic_base, critic_extra], 1)}

    # ------------------------------------------------------------------ rewards / dones
    def _update_flags(self):
        pos, quat, vel = self._obj_state()
        grasp, tips, palm, wrist = self._hand_geom()
        rise = pos[:, 2] - self.rest_z
        ph = self._phase()
        lifted = rise >= LIFT_M
        self.lifted_any |= lifted
        in_hold = ph == 3
        self.held_hold &= ~in_hold | lifted
        near_counter = (tips[:, :, 2] < self.counter_z + 0.003).any(1)
        self.counter_ticks += near_counter.float()
        self.min_obj_dz = torch.minimum(self.min_obj_dz, rise)
        # safety proxy: wrist + palm vs counter keep-out (z < 0.03 over the counter footprint) and workspace box
        viol = torch.zeros_like(lifted)
        for p in (palm, wrist):
            over = (p[:, 0] > 0.14) & (p[:, 0] < 1.20) & (p[:, 1].abs() < 0.55)
            viol |= over & (p[:, 2] < 0.03 - 0.01)
            viol |= (p[:, 0] < 0.0 + 0.01) | (p[:, 0] > 0.80 - 0.01) | (p[:, 1].abs() > 0.70 - 0.01) | \
                    (p[:, 2] < -0.25 + 0.01) | (p[:, 2] > 0.60 - 0.01)
        self.violated |= viol
        self._lifted, self._rise, self._near_counter, self._pos = lifted, rise, near_counter, pos

    def _get_dones(self):
        self._update_flags()
        L = self.lib["length"][self.entry]
        timeout = self.t >= L
        fell = self._pos[:, 2] < self.counter_z - 0.03
        self.success = timeout & self.held_hold & self._lifted & ~self.violated
        return fell, timeout

    def _get_rewards(self):
        c = self.cfg
        ph = self._phase()
        r = c.w_success * self.success.float()
        r += c.w_lift * torch.clamp(self._rise / LIFT_M, 0.0, 1.0) * (ph >= 1).float()
        r -= c.w_action_rate * ((self.res_a - self.prev_res_a) ** 2).sum(1) / self.cfg.action_ema ** 2
        r -= c.w_counter * self._near_counter.float()
        r -= c.w_violation * self.violated.float() * (self.t >= self.lib["length"][self.entry]).float()
        self.extras.setdefault("log", {})
        return r

    # ------------------------------------------------------------------ reset
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        done = env_ids[self.t[env_ids] > 0]
        if len(done):
            log = self.extras.setdefault("log", {})
            log["Episode/success"] = self.success[done].float().mean()
            log["Episode/lifted_any"] = self.lifted_any[done].float().mean()
            log["Episode/violation"] = self.violated[done].float().mean()
            log["Episode/counter_ticks"] = self.counter_ticks[done].mean()
            if self.collect_stats:
                self.ep_stats.append({k: v[done].cpu() for k, v in dict(
                    env_id=torch.arange(self.num_envs, device=self.device), t=self.t,
                    entry=self.entry, est_shape=self.lib["shape"][self.entry], shape=self.true_shape,
                    success=self.success, lifted_any=self.lifted_any, held_hold=self.held_hold,
                    violated=self.violated, counter_ticks=self.counter_ticks, min_obj_dz=self.min_obj_dz,
                    final_rise=self._rise if hasattr(self, "_rise") else torch.zeros_like(self.rest_z)).items()})
        super()._reset_idx(env_ids)
        self._reset_episodes(env_ids)

    def _reset_episodes(self, env_ids, entries=None, true_shape=None, pose_noise=None):
        c, dev, n = self.cfg, self.device, len(env_ids)
        e = torch.randint(0, self.n_lib, (n,), device=dev) if entries is None else entries.to(dev)
        self.entry[env_ids] = e
        shp = self.lib["shape"][e].clone()
        if true_shape is not None:
            shp = true_shape.to(dev)
        elif c.est_wrong_shape_p > 0:
            wrong = torch.rand(n, device=dev) < c.est_wrong_shape_p
            shp = torch.where(wrong, (shp + torch.randint(1, 3, (n,), device=dev)) % 3, shp)
        self.true_shape[env_ids] = shp
        if pose_noise is None:
            dxy = (torch.rand(n, 2, device=dev) * 2 - 1) * c.est_pos_noise
            dyaw = (torch.rand(n, device=dev) * 2 - 1) * math.radians(c.est_yaw_noise_deg)
        else:
            dxy, dyaw = pose_noise[0].to(dev), pose_noise[1].to(dev)
        x = self.lib["x"][e] + dxy[:, 0]
        y = self.lib["y"][e] + dxy[:, 1]
        yaw = self.lib["yaw_pelvis"][e] + dyaw
        org = self.scene.env_origins[env_ids] + self.pelvis_w
        zero6 = torch.zeros(n, 6, device=dev)
        for i, (s, o) in enumerate(zip(SHAPES, self.objs)):
            act = shp == i
            pz = self.counter_z + self.obj_h[s] / 2 + 0.001
            px, py = _park_xy(i)
            p = torch.stack([torch.where(act, x, torch.full_like(x, px)),
                             torch.where(act, y, torch.full_like(y, py)),
                             torch.where(act, torch.full_like(x, pz), torch.full_like(x, -PELVIS_W[2] + 0.03))], 1)
            yy = torch.where(act, yaw, torch.zeros_like(yaw))
            q = torch.stack([torch.zeros_like(yy), torch.zeros_like(yy), torch.sin(yy / 2), torch.cos(yy / 2)], 1)
            o.write_root_pose_to_sim_index(root_pose=torch.cat([p + org, q], 1), env_ids=env_ids)
            o.write_root_velocity_to_sim_index(root_velocity=zero6, env_ids=env_ids)
        self.rest_z[env_ids] = self.counter_z + torch.tensor([self.obj_h[s] for s in SHAPES], device=dev)[shp] / 2
        # robot: legs/waist default, left arm at the pregrasp, right arm home, hands at the start posture
        q = self.default_q[env_ids].clone()
        q[:, self.arm_ids] = self.lib["start_arm"][e]
        q[:, self.hand_ids] = self.lib["start_hand"][e]
        self.robot.write_joint_position_to_sim_index(position=q, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q), env_ids=env_ids)
        self.target[env_ids] = q[:, self.ctrl_ids]
        self.robot.actuators.target_command.set_position_index(value=q, env_ids=env_ids)
        self.t[env_ids] = 0
        self.actions[env_ids] = 0
        self.prev_actions[env_ids] = 0
        self.res_a[env_ids] = 0
        self.prev_res_a[env_ids] = 0
        self.held_hold[env_ids] = True
        self.violated[env_ids] = False
        self.lifted_any[env_ids] = False
        self.success[env_ids] = False
        self.counter_ticks[env_ids] = 0
        self.min_obj_dz[env_ids] = 0
