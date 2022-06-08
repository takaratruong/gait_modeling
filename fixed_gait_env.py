import numpy as np

from gym import utils
import my_mujoco_env

from scipy.interpolate import interp1d
from stable_baselines3 import PPO
from my_pd_walker import PD_Walker2dEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

import torch

import ipdb

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -20.0,
}

class FixedGait(my_mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(
            self,
            xml_file="walker2d.xml",
            forward_reward_weight=1.0,
            ctrl_cost_weight=1e-3,
            healthy_reward=1.0,
            terminate_when_unhealthy=True,
            healthy_z_range=(0.6, 2.0),
            healthy_angle_range=(-1., 1.),
            reset_noise_scale=5e-3,
            exclude_current_positions_from_observation=True,
            args=None
    ):
        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy

        self._healthy_z_range = healthy_z_range
        self._healthy_angle_range = healthy_angle_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        # My variables:
        self.args = args

        self.fixed_gait_policy = args.fixed_gait_policy

        # PD
        self.p_gain = 100
        self.d_gain = 10

        # Gait Reference
        ref = np.loadtxt('2d_walking.txt')
        self.gait_ref = interp1d(np.arange(0, 11) / 10, ref, axis=0)
        self.gait_vel_ref = interp1d(np.arange(0, 11) / 10,
                                     np.diff(np.concatenate((np.zeros((1, 9)), ref)) / .1, axis=0), axis=0)

        self._phase_cntr = 0
        self._max_phase = 500  # 1/.002

        self.max_ep_time = 20  # seconds
        self.total_reward = 0

        self.gait_cycle_time = 1.0
        self.time_offset = np.random.uniform(0, self.gait_cycle_time)  # offsets gait cycle time

        # impulse
        self.impulse_duration = .2
        self.impulse_delay = 5  # 3.5+ np.random.uniform(0, 1)  # time between impulses
        self.impulse_time_start = 0  # used to keep track of elapsed time since impulse
        self.force = self.args.f
        self.lower_impact_lim = 70.0

        my_mujoco_env.MujocoEnv.__init__(self, xml_file, 12)
        self.init_qvel[0] = 1
        self.init_qvel[1:] = 0
        self.target_ref = self.init_qpos

    @property
    def healthy_reward(self):
        return (
                float(self.is_healthy or self._terminate_when_unhealthy)
                * self._healthy_reward
        )

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def is_healthy(self):
        z, angle = self.sim.data.qpos[1:3]

        min_z, max_z = self._healthy_z_range
        min_angle, max_angle = self._healthy_angle_range

        healthy_z = min_z < z < max_z
        healthy_angle = min_angle < angle < max_angle
        is_healthy = healthy_z and healthy_angle

        if self.sim.data.time > self.max_ep_time:
            is_healthy = False
        return is_healthy

    # For vec env
    def reset_time_limit(self):
        if self.sim.data.time > self.max_ep_time:
            return self.reset()
        else:
            return self._get_obs()

    @property
    def done(self):
        done = not self.is_healthy if self._terminate_when_unhealthy else False
        return done

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = np.clip(self.sim.data.qvel.flat.copy(), -1000, 1000)

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        if self.cntr2phase(self._phase_cntr < .5):
            observation = np.concatenate((position, velocity / 10, np.array(
                [self.cntr2phase(self._phase_cntr)]))).ravel()  # , self.target_ref[0] - self.sim.data.qpos[0]
        else:
            observation = np.concatenate((self.sim.data.qpos[1:3], self.sim.data.qpos[6:9], self.sim.data.qpos[3:6],
                                          self.sim.data.qvel[0:3] / 10,
                                          self.sim.data.qvel[6:9] / 10, self.sim.data.qvel[3:6] / 10,
                                          np.array([self.cntr2phase(
                                              self._phase_cntr) - 0.5])))  # , self.target_ref[0] - self.sim.data.qpos[0]]

        return observation

    def observe(self):
        return self._get_obs()

    def get_total_reward(self):
        return self.total_reward

    def step(self, action):
        new_action = np.array(action.tolist())

        fixed_gait_action, _ = self.fixed_gait_policy.predict(self._get_obs())
        fixed_gait_action = np.array(fixed_gait_action.tolist())

        action = new_action + fixed_gait_action

        if self.cntr2phase(self._phase_cntr) < 0.5:
            joint_action = action[0:6].copy()
        else:
            joint_action = action[[3, 4, 5, 0, 1, 2]].copy()

        phase_action = int(50 * action[6])  # 50

        self._phase_cntr += phase_action

        target_phase_cntr = self._phase_cntr + self.frame_skip

        target_ref = self.get_pos_ref(self.cntr2phase(target_phase_cntr))

        joint_target = joint_action + target_ref[3:]  # add action to joint ref to create final joint target

        for _ in range(self.frame_skip):

            joint_obs = self.sim.data.qpos[3:]
            joint_vel_obs = self.sim.data.qvel[3:]

            error = joint_target - joint_obs
            error_der = joint_vel_obs

            torque = self.p_gain * error - self.d_gain * error_der

            self.do_simulation(torque / 100, 1)
            self._phase_cntr += 1

            # Apply force for specified duration
            if self.args.imp or self.args.c_imp:
                if (self.data.time - self.impulse_time_start) > self.impulse_delay:
                    self.sim.data.qfrc_applied[0] = self.force

                if (self.data.time - self.impulse_time_start - self.impulse_delay) >= self.impulse_duration:
                    self.sim.data.qfrc_applied[0] = 0
                    self.impulse_time_start = self.data.time
                    self.force = self.args.f if self.args.c_imp else self.get_rand_force()

        # Calculate Reward based on target and mujocos simulator
        joint_ref = target_ref[3:]
        joint_obs = self.sim.data.qpos[3:9]
        joint_reward = np.exp(-2 * np.sum((joint_ref - joint_obs) ** 2))

        pos_ref = target_ref[0:2]
        pos = self.sim.data.qpos[0:2]
        pos_reward = 10 * (1 - self.sim.data.qvel[0]) ** 2 + (pos_ref[1] - pos[1]) ** 2
        pos_reward = np.exp(-pos_reward)

        orient_ref = target_ref[2]
        orient_obs = self.sim.data.qpos[2]
        orient_reward = 2 * ((orient_ref - orient_obs) ** 2) + 5 * (self.sim.data.qvel[2] ** 2)
        orient_reward = np.exp(-orient_reward)

        reward = 0.3 * orient_reward + 0.4 * joint_reward + 0.3 * pos_reward

        self.total_reward += reward

        observation = self._get_obs()

        done = self.done
        info = {}
        if self.sim.data.time >= self.max_ep_time:
            info["TimeLimit.truncated"] = True
        else:
            info["TimeLimit.truncated"] = False

        return observation, reward, done, info

    def cntr2phase(self, phase_cntr):
        if phase_cntr >= 0:
            return (phase_cntr % self._max_phase) / self._max_phase
        else:
            return ((self._max_phase + phase_cntr) % self._max_phase) / self._max_phase

    def get_pos_ref(self, phase):
        gait_ref = self.gait_ref(phase)
        gait_ref[0] += np.floor(phase)
        gait_ref[1] += 1.25
        return gait_ref

    def get_vel_ref(self, phase):
        return self.gait_vel_ref(phase)

    def get_rand_force(self):
        neg_force = np.random.randint(-abs(self.args.f), -abs(self.lower_impact_lim))
        pos_force = np.random.randint(abs(self.lower_impact_lim), abs(self.args.f))
        return neg_force if np.random.rand() <= .5 else pos_force

    def reset_model(self):
        # randomize starting phase alignment
        offset = int(np.random.randint(0, self._max_phase / 10) * 10)

        self._phase_cntr = offset
        init_pos_ref = self.get_pos_ref(self.cntr2phase(self._phase_cntr))
        init_vel_ref = self.get_pos_ref(self.cntr2phase(self._phase_cntr))

        self.impulse_delay = 5  # + np.random.uniform(0, 2)
        self.impulse_time_start = self.data.time + .01

        self.set_state(init_pos_ref, self.init_qvel)
        self.total_reward = 0

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)


