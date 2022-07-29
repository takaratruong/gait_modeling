import numpy as np
from gym import utils
from scipy.interpolate import interp1d
import ipdb
import os
from scipy.spatial.transform import Rotation as R
import time

import environments.humanoid.humanoid_mujoco_env as mujoco_env_humanoid
from environments.humanoid.humanoid_utils import flip_action, flip_position, flip_velocity

class HumanoidTreadmillEnv(mujoco_env_humanoid.MujocoEnv, utils.EzPickle):

    def __init__(
            self,
            xml_file="humanoid_treadmill.xml",
            terminate_when_unhealthy=True,
            healthy_z_range=(1, 10.0), #.7
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
        ref = np.loadtxt(args.gait_ref_file)
        # ref = np.loadtxt('../environments/humanoid/humanoid_walk_ref.txt')
        # ref[:, 2] = 1.2 # add treadmill offset to z height if not done so already
        # ref = np.hstack((ref, np.zeros(ref.shape[0]).reshape(-1,1)))

        self.gait_ref = interp1d(np.arange(0, ref.shape[0]) / (ref.shape[0]-1), ref, axis=0)
        self.treadmill_velocity = args.treadmill_velocity
        self.gait_cycle_time = args.gait_cycle_time
        self.time_step = .01

        self.initial_phase_offset = np.random.randint(0, 50) / 50

        self.action_phase_offset = 0

        self.max_ep_time = self.args.max_ep_time
        self.frame_skip = self.args.frame_skip

        self.total_reward = 0

        self.force = self.get_force()

        mujoco_env_humanoid.MujocoEnv.__init__(self, xml_file, self.frame_skip)
        self.set_state(self.gait_ref(self.initial_phase_offset), self.init_qvel)

    @property
    def is_healthy(self):
        min_z, max_z = self._healthy_z_range
        is_healthy = min_z < self.data.qpos[2] < max_z

        return is_healthy

    def calc_reward(self, ref):
        joint_ref = ref[7:-1]
        joint_obs = self.sim.data.qpos[7:-1]
        joint_reward = np.exp(-2 * np.sum((joint_ref - joint_obs) ** 2))

        pos_ref = ref[0:3]
        pos = self.sim.data.qpos[0:3]
        pos_reward = 10 * (0 - self.sim.data.qvel[0]) ** 2 + \
                     2 * (0 - np.sum(self.sim.data.get_body_xvelp('neck')[0])) ** 2 + \
                     1 * (np.sum(self.sim.data.get_body_xvelp('neck')[1])) ** 2 + \
                     1 * (pos_ref[1] - pos[1]) ** 2 + (pos_ref[2] - pos[2]) ** 2

        pos_reward = np.exp(-pos_reward)

        orient_ref = ref[3:7]
        orient_obs = self.sim.data.qpos[3:7]
        orient_reward = 2 * np.sum((orient_ref - orient_obs) ** 2) + 5 * np.sum(
            (self.sim.data.qvel[3:6]) ** 2)  # <-- fix later
        orient_reward = np.exp(-orient_reward)

        # reward = self.args.rot_weight * orient_reward + self.args.jnt_weight * joint_reward + self.args.pos_weight * pos_reward
        reward = orient_reward * joint_reward * pos_reward

        return reward

    @property
    def done(self):
        done = not self.is_healthy if self._terminate_when_unhealthy else False
        return done

    def _get_obs(self):
        if self.phase <= .5:
            position = self.sim.data.qpos.flat.copy()
            position = position[:-1]
            velocity = np.clip(self.sim.data.qvel.flat.copy()[:-1], -1000, 1000)
            observation = np.concatenate((position, velocity / 10, np.array([self.phase]))).ravel()

        else:
            position = self.sim.data.qpos.flat.copy()
            position = position[:-1]
            flipped_pos = flip_position(position)

            velocity = np.clip(self.sim.data.qvel.flat.copy()[:-1], -1000, 1000)
            flipped_vel = flip_velocity(velocity)

            observation = np.concatenate((flipped_pos, flipped_vel / 10, np.array([self.phase - .5]))).ravel()
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

        return gait_ref

    def get_force(self):
        force = self.args.perturbation_force
        dir = self.args.perturbation_dir
        if self.args.rand_perturbation:
            neg_force = np.random.randint(-abs(self.args.perturbation_force), -abs(self.args.min_perturbation_force_mag))
            pos_force = np.random.randint(abs(self.args.min_perturbation_force_mag), abs(self.args.perturbation_force))
            force = neg_force if np.random.rand() <= .5 else pos_force
            dir = np.random.randint(2)
        return (dir, force)

    def apply_force(self, force):
        dir = force[0]
        mag = force[1]
        self.sim.data.qfrc_applied[dir] = mag

    def zero_applied_force(self):
        self.sim.data.qfrc_applied[0] = 0
        self.sim.data.qfrc_applied[1] = 0

    def step(self, action):
        # print(len(action))
        # ipdb.set_trace()
        action = np.array(action.tolist()).copy()

        joint_action = action[:-1] if self.phase <= .5 else flip_action(action[:-1])

        phase_action = self.args.phase_action_mag * action[-1]

        self.action_phase_offset += phase_action

        target_ref = self.target_reference
        final_target = joint_action + target_ref[7:-1]  # add action to joint ref to create final joint target

        for _ in range(self.frame_skip):
            joint_obs = self.sim.data.qpos[7:-1].copy()
            joint_vel_obs = self.sim.data.qvel[6:-1].copy()

            error = final_target - joint_obs
            error_der = joint_vel_obs

            torque = 100 * error - 10 * error_der

            self.sim.data.set_joint_qvel('treadmill', -self.treadmill_velocity)  # keep treadmill moving

            self.do_simulation(torque / 100, 1)

            # self.set_state(target_ref, self.init_qvel)

        observation = self._get_obs()

        reward = self.calc_reward(target_ref)
        self.total_reward += reward

        done = self.done

        info = {}
        if self.data.time >= self.max_ep_time:
            info["TimeLimit.truncated"] = not done
            done = True

        return observation, reward, done, info

    def reset_model(self):
        self.action_phase_offset = 0
        self.initial_phase_offset = np.random.randint(0, 50) / 50

        qpos = self.gait_ref(self.phase)
        self.set_state(qpos, self.init_qvel)

        self.total_reward = 0
        self.force = self.get_force()
        observation = self._get_obs()

        return observation

    def reset_time_limit(self):
        if self.sim.data.time > self.max_ep_time:
            return self.reset()
        else:
            return self._get_obs()

    def get_total_reward(self):
        return self.total_reward

    def get_elapsed_time(self):
        return self.data.time