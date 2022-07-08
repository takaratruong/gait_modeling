import numpy as np
from gym import utils
from scipy.interpolate import interp1d
import ipdb
import os

import environments.humanoid.humanoid_mujoco_env as mujoco_env_humanoid

"""


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 3,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -20.0,
}
"""

class Humanoid_test_env2(mujoco_env_humanoid.MujocoEnv, utils.EzPickle):

    def __init__(
            self,
            xml_file="humanoid.xml",
            terminate_when_unhealthy=True,
            healthy_z_range=(0.7, 10.0),
            args=None
    ):
        utils.EzPickle.__init__(**locals())
        self._healthy_z_range = healthy_z_range
        self._terminate_when_unhealthy = terminate_when_unhealthy

        # My variables:
        self.args = args

        # PD
        self.p_gain = 100
        self.d_gain = 10

        # Interpolation of gait reference wrt phase.
        ref = np.loadtxt('environments/humanoid/humanoid_walk_ref.txt')
        self.gait_ref = interp1d(np.arange(0, 39) / 38, ref, axis=0)

        self.gait_cycle_time = 38 * 0.03333200
        self.time_step = .0166

        self.initial_phase_offset = np.random.randint(0, 50) / 50

        self.action_phase_offset = 0

        self.max_ep_time = self.args.max_ep_time

        self.frame_skip = self.args.frame_skip

        self.total_reward = 0

        mujoco_env_humanoid.MujocoEnv.__init__(self, xml_file, self.frame_skip)

        self.init_qvel[0] = 1
        self.init_qvel[1:] = 0

        self.set_state(self.gait_ref(self.initial_phase_offset), self.init_qvel)

    @property
    def is_healthy(self):
        min_z, max_z = self._healthy_z_range
        is_healthy = min_z < self.data.qpos[2] < max_z
        return is_healthy

    def calc_reward(self, obs, ref):
        joint_ref = ref[7:]
        joint_obs = self.sim.data.qpos[7:35]
        joint_reward = np.exp(-2 * np.sum((joint_ref - joint_obs) ** 2))

        pos_ref = ref[0:3]
        pos = self.sim.data.qpos[0:3]
        pos_reward = 10 * ( 1.23859/self.gait_cycle_time  - self.sim.data.qvel[0]) ** 2 + (pos_ref[1] - pos[1]) ** 2 + (pos_ref[2] - pos[2]) ** 2
        pos_reward = np.exp(-pos_reward)

        orient_ref = ref[3:7]
        orient_obs = self.sim.data.qpos[3:7]
        orient_reward = 2 * np.sum((orient_ref - orient_obs) ** 2) + 5 * np.sum((self.sim.data.qvel[3:6]) ** 2)
        orient_reward = np.exp(-orient_reward)

        reward = self.args.orient_weight * orient_reward + self.args.joint_weight * joint_reward + self.args.pos_weight * pos_reward

        return reward

    @property
    def done(self):
        done = not self.is_healthy if self._terminate_when_unhealthy else False
        return done

    def _get_obs(self):
        if self.phase < .5:
            position = self.sim.data.qpos.flat.copy()
            position = position[1:]
            velocity = np.clip(self.sim.data.qvel.flat.copy(), -1000, 1000)
            observation = np.concatenate((position, velocity / 10, np.array([self.phase]))).ravel()

        else:
            position = self.sim.data.qpos.flat.copy()
            position = position[1:]

            velocity = np.clip(self.sim.data.qvel.flat.copy(), -1000, 1000)

            flipped_pos = np.concatenate((
                position[0:6],  # x,y, quart

                # flip chest
                np.array([-1 * position[6]]),
                np.array([position[7]]),
                np.array([-1 * position[8]]),

                # flip neck
                np.array([-1 * position[9]]),
                np.array([position[10]]),
                np.array([-1 * position[11]]),

                # flip shoulders and elbow
                position[16:20],
                position[12:16],

                # flip hip knee and ankle
                position[27:35],
                position[20:27],
            ))

            flipped_vel = np.concatenate((
                velocity[0:3] / 10,

                # flip center rotation
                np.array([-1 * velocity[3] / 10]),
                np.array([velocity[4] / 10]),
                np.array([-1 * velocity[5] / 10]),

                # flip chest
                np.array([-1 * velocity[6] / 10]),
                np.array([velocity[7] / 10]),
                np.array([-1 * velocity[8] / 10]),

                # flip neck
                np.array([-velocity[9] / 10]),
                np.array([velocity[10] / 10]),
                np.array([-velocity[11] / 10]),

                # flip shoulders and elbow
                velocity[16:20] / 10,
                velocity[12:16] / 10,

                # flip hip knee and ankle
                velocity[27:35] / 10,
                velocity[20:27] / 10,
            ))

            observation = np.concatenate((flipped_pos, flipped_vel, np.array([self.phase - 0.5])))

        return observation

    def observe(self):
        return self._get_obs()

    @property
    def phase(self):
        initial_offset_time = self.initial_phase_offset * self.gait_cycle_time
        action_offset_time = self.action_phase_offset * self.gait_cycle_time
        total_time = initial_offset_time + action_offset_time + self.data.time

        return (total_time % self.gait_cycle_time) / self.gait_cycle_time

    @property
    def target_reference(self):
        frame_skip_time = self.frame_skip * self.time_step
        initial_offset_time = self.initial_phase_offset * self.gait_cycle_time
        action_offset_time = self.action_phase_offset * self.gait_cycle_time
        total_time = frame_skip_time + initial_offset_time + action_offset_time + self.data.time

        phase_target = (total_time % self.gait_cycle_time) / self.gait_cycle_time
        gait_ref = self.gait_ref(phase_target)
        gait_ref[0] += 1.23859 * np.floor(total_time / self.gait_cycle_time)

        return gait_ref

    def get_rand_force(self):
        neg_force = np.random.randint(-abs(self.args.perturbation_force), -abs(self.args.min_perturbation_force_mag))
        pos_force = np.random.randint(abs(self.args.min_perturbation_force_mag), abs(self.args.perturbation_force))
        return neg_force if np.random.rand() <= .5 else pos_force

    def step(self, action):
        action = np.array(action.tolist())
        if self.phase < 0.5:
            joint_action = action[0:28]
        else:
            joint_action = action[[0, 1, 2, 3, 4, 5,
                                   10, 11, 12, 13,
                                   6, 7, 8, 9,
                                   21, 22, 23, 24, 25, 26, 27,
                                   14, 15, 16, 17, 18, 19, 20]]

        phase_action = self.args.phase_action_mag * action[28]

        self.action_phase_offset += phase_action

        final_target = joint_action + self.target_reference[7:]  # add action to joint ref to create final joint target

        for _ in range(self.frame_skip):
            joint_obs = self.sim.data.qpos[7:].copy()
            joint_vel_obs = self.sim.data.qvel[6:].copy()

            error = final_target - joint_obs
            error_der = joint_vel_obs

            torque = 100 * error - 10 * error_der

            self.do_simulation(torque/10, 1)

        #self.set_state(self.target_reference, self.init_qvel)

        observation = self._get_obs()
        reward = self.calc_reward(observation, self.target_reference)

        self.total_reward += reward

        done = self.done
        info = {}

        if self.data.time >= self.max_ep_time:
            info["TimeLimit.truncated"] = not done
            done = True

        return observation, reward, done, info

    def reset_model(self):
        self.total_reward = 0
        self.action_phase_offset = 0
        self.initial_phase_offset = np.random.randint(0, 50) / 50

        qpos = self.gait_ref(self.phase)

        self.set_state(qpos, self.init_qvel)

        observation = self._get_obs()
        return observation

    def reset_time_limit(self):
        if self.sim.data.time > self.max_ep_time:
            return self.reset()
        else:
            return self._get_obs()

    def get_total_reward(self):
        return self.total_reward

"""
  

        joint_ref = ref[7:]
        joint_obs = self.sim.data.qpos[7:35]
        joint_reward = np.exp(-2 * np.sum((joint_ref - joint_obs) ** 2))

        pos_ref = ref[0:3]
        pos = self.sim.data.qpos[0:3]
        pos_reward = 10 * (1 - self.sim.data.qvel[0]) ** 2 + (pos_ref[1] - pos[1]) ** 2 + (pos_ref[2] - pos[2]) ** 2
        pos_reward = np.exp(-pos_reward)
        
        orient_ref = ref[3:7]
        orient_obs = self.sim.data.qpos[3:7]
        orient_reward = 2 * np.sum((orient_ref - orient_obs) ** 2) + 5 * np.sum((self.sim.data.qvel[3:6]) ** 2)
        orient_reward = np.exp(-orient_reward)
        
"""