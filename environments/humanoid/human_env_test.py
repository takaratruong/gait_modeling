import numpy as np
from gym import utils
from scipy.interpolate import interp1d
import ipdb
import os

import environments.humanoid.humanoid_mujoco_env as mujoco_env_humanoid

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -20.0,
}

class Humanoid_test_env(mujoco_env_humanoid.MujocoEnv, utils.EzPickle):

    def __init__(
            self,
            xml_file="humanoid.xml",
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
        ref = np.loadtxt('environments/humanoid/humanoid_walk_ref.txt')

        self.gait_cycle_time = 1

        self.gait_ref = interp1d(np.arange(0, 39) / 38, ref, axis=0)
        #self.gait_vel_ref = interp1d(np.arange(0, 11) / 10, np.diff(np.concatenate((np.zeros((1, 9)), ref))/.1, axis=0), axis=0)

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

        mujoco_env_humanoid.MujocoEnv.__init__(self, xml_file, self.frame_skip)
        self.init_qvel[0] = 1
        self.init_qvel[1:] = 0

    @property
    def is_healthy(self):
        min_z, max_z = self._healthy_z_range

        is_healthy = min_z < self.data.qpos[2] < max_z

        return is_healthy

    @property
    def done(self):
        done = not self.is_healthy if self._terminate_when_unhealthy else False
        return done

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = np.clip(self.sim.data.qvel.flat.copy(), -1000, 1000)

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity / 10, np.array([self.phase]))).ravel()

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
        gait_ref[0] += 1.23859 * np.floor(cntr / self._max_phase) - 1.23859 * self.gait_ref((self.offset % self._max_phase)/self._max_phase)[0]
        return gait_ref

    def get_rand_force(self):
        neg_force = np.random.randint(-abs(self.args.perturbation_force), -abs(self.args.min_perturbation_force_mag))
        pos_force = np.random.randint(abs(self.args.min_perturbation_force_mag), abs(self.args.perturbation_force))
        return neg_force if np.random.rand() <= .5 else pos_force

    def step(self, action):
        action = np.array(action.tolist())
        if self.phase < 0.5:
            joint_action = action[0:28].copy()
        else:
            joint_action = action[[0,1,2,3,4,5,
                                   10,11,12,13,
                                   6,7,8,9,
                                   21,22,23,24,25,26,27,
                                   14,15,16,17,18,19,20]].copy()

        self.sim_cntr += int(self.args.phase_action_mag * action[28])

        ref = self.cntr2ref(self.sim_cntr + self.frame_skip)

        joint_target = joint_action + ref[7:]  # add action to joint ref to create final joint target

        for _ in range(self.frame_skip):
            joint_obs = self.sim.data.qpos[7:]
            joint_vel_obs = self.sim.data.qvel[6:]

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

        joint_ref = ref[7:]
        joint_obs = self.sim.data.qpos[7:35]
        joint_reward = np.exp(-2 * np.sum((joint_ref - joint_obs) ** 2))

        pos_ref = ref[0:3]
        pos = self.sim.data.qpos[0:3]
        pos_reward = 10 * (1-self.sim.data.qvel[0])**2 + (pos_ref[1] - pos[1]) ** 2 + (pos_ref[2] - pos[2]) ** 2
        pos_reward = np.exp(-pos_reward)

        orient_ref = ref[3:7]
        orient_obs = observation[2:6]
        orient_reward = 2 * np.sum( (orient_ref - orient_obs) ** 2) + 5 * np.sum((self.sim.data.qvel[3:6]) ** 2)
        orient_reward = np.exp(-orient_reward)

        reward = 0.3*orient_reward+0.4*joint_reward+0.3*pos_reward

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



"""
 else:
            observation = np.concatenate((
                # ignore rotation for now <-- quarternion
                self.sim.data.qpos[1:7],

                # flip chest
                np.array([-1*self.sim.data.qpos[7]]),
                np.array([self.sim.data.qpos[8]]),
                np.array([-1*self.sim.data.qpos[9]]),

                #flip neck
                np.array([-1*self.sim.data.qpos[10]]),
                np.array([self.sim.data.qpos[11]]),
                np.array([-1*self.sim.data.qpos[12]]),

                #flip shoulders and elbow
                self.sim.data.qpos[17:21],
                self.sim.data.qpos[13:17],

                #flip hip knee and ankle
                self.sim.data.qpos[28:35],
                self.sim.data.qpos[21:28],

                # Velocities ###################
                self.sim.data.qvel[0:3]/10,

                # flip center rotation
                np.array([-1*self.sim.data.qvel[3]/10]),
                np.array([self.sim.data.qvel[4]/10]),
                np.array([-1*self.sim.data.qvel[5]/10]),

                # flip chest
                np.array([-1*self.sim.data.qpos[6]/10]),
                np.array([self.sim.data.qpos[7]/10]),
                np.array([-1*self.sim.data.qpos[8]/10]),

                # flip neck
                np.array([-self.sim.data.qpos[9]/10]),
                np.array([self.sim.data.qpos[10]/10]),
                np.array([-self.sim.data.qpos[11]/10]),

                # flip shoulders and elbow
                self.sim.data.qpos[16:20]/10,
                self.sim.data.qpos[12:16]/10,

                # flip hip knee and ankle
                self.sim.data.qpos[20:27]/10,
                self.sim.data.qpos[27:35]/10,

                np.array([self.phase - 0.5])
            ))#.ravel()

            #observation = np.concatenate((self.sim.data.qpos[1:3], self.sim.data.qpos[6:9], self.sim.data.qpos[3:6], self.sim.data.qvel[0:3]/10,
            #    self.sim.data.qvel[6:9]/10, self.sim.data.qvel[3:6]/10, np.array([self.phase - 0.5])))

"""