import numpy as np
from gym import utils
from scipy.interpolate import interp1d
import ipdb
import os

import environments.walker2d.walker2d_mujoco_env as walker2d_mujoco_env

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -20.0,
}

class WalkerEnv(walker2d_mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(
            self,
            xml_file="walker2d.xml",
            forward_reward_weight=1.0,
            ctrl_cost_weight=1e-3,
            healthy_reward=1.0,
            terminate_when_unhealthy=True,
            healthy_z_range=(0.8, 2.0),
            healthy_angle_range=(-1.0, 1.0),
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

        # PD
        self.p_gain = 100
        self.d_gain = 10

        # Interpolation of gait reference wrt phase.
        ref = np.loadtxt('environments/walker2d/2d_walking.txt')

        self.gait_cycle_time = 1
        self.gait_ref = interp1d(np.arange(0, 11) / 10, ref, axis=0)
        self.gait_vel_ref = interp1d(np.arange(0, 11) / 10, np.diff(np.concatenate((np.zeros((1, 9)), ref))/.1, axis=0), axis=0)

        self._max_phase = 500  # 1/.002

        self.sim_cntr = 0
        self.offset = int(np.random.randint(0, self._max_phase / 10) * 10)

        self.max_ep_time = self.args.max_ep_time
        self.max_ep_steps = self.args.num_steps
        self.elapsed_steps = 0

        # Impulse
        self.force = 0
        self.impulse_duration = self.args.perturbation_duration
        self.impulse_delay = self.args.perturbation_delay
        self.impulse_time_start = 0

        self.frame_skip = self.args.frame_skip

        # for amp
        self.total_reward = 0

        walker2d_mujoco_env.MujocoEnv.__init__(self, xml_file, self.frame_skip)
        self.init_qvel[0] = 1
        self.init_qvel[1:] = 0

    @property
    def is_healthy(self):
        z, angle = self.sim.data.qpos[1:3]

        min_z, max_z = self._healthy_z_range
        min_angle, max_angle = self._healthy_angle_range

        healthy_z = min_z < z < max_z
        healthy_angle = min_angle < angle < max_angle
        is_healthy = healthy_z and healthy_angle

        return is_healthy

    @property
    def done(self):
        done = not self.is_healthy if self._terminate_when_unhealthy else False
        return done

    def _get_obs(self):
        if self.phase < 0.5:
            position = self.sim.data.qpos.flat.copy()
            #ipdb.set_trace()
            velocity = np.clip(self.sim.data.qvel.flat.copy(), -1000, 1000)

            if self._exclude_current_positions_from_observation:
                position = position[1:]

            observation = np.concatenate((position, velocity / 10, np.array([self.phase]))).ravel()

        else:
            observation = np.concatenate((self.sim.data.qpos[1:3], self.sim.data.qpos[6:9], self.sim.data.qpos[3:6], self.sim.data.qvel[0:3]/10,
                self.sim.data.qvel[6:9]/10, self.sim.data.qvel[3:6]/10, np.array([self.phase - 0.5])))

        return observation

    def observe(self):
        return self._get_obs()

    @property
    def phase(self):
        return self.cntr2phase(self.sim_cntr)

    def cntr2phase(self, cntr):
        if cntr >= 0:
            return ((cntr + self.offset) % self._max_phase) / self._max_phase
        else:
            return ((self._max_phase + cntr + self.offset) % self._max_phase) / self._max_phase

    def cntr2ref(self, cntr):
        phase = self.cntr2phase(cntr)
        gait_ref = self.gait_ref(phase)
        gait_ref[0] += np.floor(cntr / self._max_phase) - self.gait_ref((self.offset % self._max_phase)/self._max_phase)[0]
        gait_ref[1] += 1.25
        return gait_ref

    def get_rand_force(self):
        neg_force = np.random.randint(-abs(self.args.perturbation_force), -abs(self.args.min_perturbation_force_mag))
        pos_force = np.random.randint(abs(self.args.min_perturbation_force_mag), abs(self.args.perturbation_force))
        return neg_force if np.random.rand() <= .5 else pos_force

    def step(self, action):
        action = np.array(action.tolist())
        if self.phase < 0.5:
            joint_action = action[0:6].copy()
        else:
            joint_action = action[[3, 4, 5, 0, 1, 2]].copy()

        self.sim_cntr += int(self.args.phase_action_mag * action[6]*0)

        ref = self.cntr2ref(self.sim_cntr + self.frame_skip)
        joint_target = joint_action + ref[3:]  # add action to joint ref to create final joint target

        for _ in range(self.frame_skip):
            joint_obs = self.sim.data.qpos[3:]
            joint_vel_obs = self.sim.data.qvel[3:]

            error = joint_target - joint_obs
            error_der = joint_vel_obs

            torque = self.p_gain * error - self.d_gain * error_der

            self.do_simulation(torque/100, 1)
            self.sim_cntr += 1

            # Apply force for specified duration
            if self.args.const_perturbation or self.args.rand_perturbation:
                if (self.data.time - self.impulse_time_start) > self.impulse_delay:
                    self.sim.data.qfrc_applied[0] = self.force

                if (self.data.time - self.impulse_time_start - self.impulse_delay) >= self.impulse_duration:
                    self.sim.data.qfrc_applied[0] = 0
                    self.impulse_time_start = self.data.time
                    self.force = self.args.perturbation_force if self.args.const_perturbation else self.get_rand_force()

        observation = self._get_obs()

        joint_ref = ref[3:]
        joint_obs = self.sim.data.qpos[3:9]
        joint_reward = np.exp(-2 * np.sum((joint_ref - joint_obs) ** 2))

        pos_ref = ref[0:2]
        pos = self.sim.data.qpos[0:2]
        pos_reward = 10 * (1-self.sim.data.qvel[0])**2 + (pos_ref[1] - pos[1] ) ** 2
        pos_reward = np.exp(-pos_reward)

        orient_ref = ref[2]
        orient_obs = observation[1]
        orient_reward = 2 * ((orient_ref - orient_obs) ** 2) + 5 * (self.sim.data.qvel[2] ** 2)
        orient_reward = np.exp(-orient_reward)

        reward = 0.3*orient_reward+0.4*joint_reward+0.3*pos_reward
        # print(orient_reward, joint_reward, pos_reward, orient_ref - orient_obs, self.sim.data.qvel[2])

        #for amp
        self.total_reward += reward

        done = self.done
        info = {}

        self.elapsed_steps+=1
        if self.elapsed_steps >= self.max_ep_steps:
            info["TimeLimit.truncated"] = not done
            done = True

        return observation, reward, done, info

    def reset_model(self):
        self.elapsed_steps = 0
        self.sim_cntr = 0
        self.offset = int(np.random.randint(0, self._max_phase / 10) * 10)
        qpos = self.cntr2ref(self.sim_cntr)

        self.impulse_delay = 5 + np.random.uniform(0, self.gait_cycle_time)
        self.impulse_time_start = self.data.time + .001

        self.set_state(qpos, self.init_qvel)
        # print("init vel", self.init_qvel)

        #for amp
        self.total_reward = 0

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

    # for amp
    def reset_time_limit(self):
        if self.sim.data.time > self.max_ep_time:
            return self.reset()
        else:
            return self._get_obs()

    def get_total_reward(self):
        return self.total_reward