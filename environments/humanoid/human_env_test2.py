import numpy as np
from gym import utils
from scipy.interpolate import interp1d
import ipdb
import os
from scipy.spatial.transform import Rotation as R
import mujoco_py
import time



import environments.humanoid.humanoid_mujoco_env as mujoco_env_humanoid
from environments.humanoid.humanoid_utils import flip_action, flip_position, flip_velocity


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

        # force application
        self.force = self.get_force()

        # MIDSTANCE STUFF
        self.midstance_impact_time = -1
        self.midstance_flag = False
        self.midstance_flag2 = False
        self.midstance_impact_applied = False
        self.record_once_flag = False
        self.post_impact_left_ankle =None
        self.post_impact_right_ankle =None

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
        pos_reward = 10 * (1.23859 / self.gait_cycle_time - self.sim.data.qvel[0]) ** 2 + 2 * (
                    1.23859 / self.gait_cycle_time - np.sum(self.sim.data.get_body_xvelp('neck')[0])) ** 2 + (
                                 pos_ref[1] - pos[1]) ** 2 + (pos_ref[2] - pos[2]) ** 2
        pos_reward = np.exp(-pos_reward)

        orient_ref = ref[3:7]
        orient_obs = obs[2:6]
        orient_reward = 2 * np.sum((orient_ref - orient_obs) ** 2) + 5 * np.sum(
            (self.sim.data.qvel[3:6]) ** 2)  # <-- fix later
        orient_reward = np.exp(-orient_reward)

        reward = self.args.orient_weight * orient_reward + self.args.joint_weight * joint_reward + self.args.pos_weight * pos_reward

        return reward

    @property
    def done(self):
        done = not self.is_healthy if self._terminate_when_unhealthy else False
        return done

    def _get_obs(self):
        if self.phase <= .5:
            position = self.sim.data.qpos.flat.copy()
            position = position[1:]
            velocity = np.clip(self.sim.data.qvel.flat.copy(), -1000, 1000)
            observation = np.concatenate((position, velocity / 10, np.array([self.phase]))).ravel()

        else:
            position = self.sim.data.qpos.flat.copy()
            position = position[1:]
            flipped_pos = flip_position(position)

            velocity = np.clip(self.sim.data.qvel.flat.copy(), -1000, 1000)
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
        gait_ref[0] += 1.23859 * np.floor(total_time / self.gait_cycle_time)

        return gait_ref

    def get_force(self):
        force = self.args.perturbation_force
        dir = self.args.perturbation_dir
        if self.args.rand_perturbation:
            neg_force = np.random.randint(-abs(self.args.perturbation_force),
                                          -abs(self.args.min_perturbation_force_mag))
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

    def check_midstance(self):
        left_foot_x = self.sim.data.get_body_xpos('left_ankle')[0]
        right_foot_x = self.sim.data.get_body_xpos('right_ankle')[0]
        right_foot_xdot = self.sim.data.get_body_xvelp('right_ankle')[0]


        #print(left_foot_x, right_foot_x)
        mid_val = False
        if right_foot_x >= left_foot_x and right_foot_xdot >= 0 and self.midstance_flag is False:
            mid_val = True
            self.midstance_impact_time = self.data.time
            self.force = self.get_force()
            self.midstance_flag = True


        #print()

        return mid_val

    def step(self, action):
        action = np.array(action.tolist()).copy()

        joint_action = action[0:28] if self.phase <= .5 else flip_action(action[0:28])

        phase_action = self.args.phase_action_mag * action[28]

        self.action_phase_offset += phase_action

        final_target = joint_action + self.target_reference[7:]  # add action to joint ref to create final joint target

        for _ in range(self.frame_skip):
            joint_obs = self.sim.data.qpos[7:].copy()
            joint_vel_obs = self.sim.data.qvel[6:].copy()

            error = final_target - joint_obs
            error_der = joint_vel_obs

            torque = 100 * error - 10 * error_der

            self.do_simulation(torque / 10, 1)

            if self.args.sim_perturbation:
                impact_timing = self.data.time % self.args.perturbation_delay
                if impact_timing >= 0 and impact_timing <= self.args.perturbation_duration and self.data.time >= self.args.perturbation_delay:
                    self.apply_force(self.force)
                else:
                    self.zero_applied_force()
                    self.force = self.get_force()
            #print(self.data.time)

            if self.args.midstance_perturbation:
                if self.data.time > self.args.perturbation_delay:

                    if self.data.get_body_xpos('right_ankle')[0] < self.data.get_body_xpos('left_ankle')[0]:
                        self.midstance_flag2 = True

                    if self.midstance_flag2:
                        self.check_midstance()

                    if self.data.time <= self.midstance_impact_time + self.args.perturbation_duration:
                        self.apply_force(self.force)
                        self.midstance_impact_applied = True
                    else:
                        self.zero_applied_force()

            if self.midstance_impact_applied:
                for i in range(self.data.ncon):
                    contact = self.data.contact[i]

                    if self.model.geom_id2name(contact.geom2) =="right_ankle":

                        if self.record_once_flag is False:
                            #ipdb.set_trace()

                            self.record_once_flag = True
                            self.post_impact_left_ankle = self.data.get_body_xpos('left_ankle')[0:2].copy()
                            self.post_impact_right_ankle = self.data.get_body_xpos('right_ankle')[0:2].copy()

        # self.set_state(self.target_reference, self.init_qvel)

        observation = self._get_obs()

        # print(self.sim.data.get_body_xvelp('neck'))
        reward = self.calc_reward(observation, self.target_reference)
        self.total_reward += reward

        done = self.done
        info = {}
        if done:
            info['is_healthy'] = self.is_healthy
            info['left_ankle'] = self.post_impact_left_ankle
            info['right_ankle'] = self.post_impact_right_ankle

        if self.data.time >= self.max_ep_time:
            info['is_healthy'] = self.is_healthy
            info['left_ankle'] = self.post_impact_left_ankle
            info['right_ankle'] = self.post_impact_right_ankle
        #    info["TimeLimit.truncated"] = not done
            done = True

        return observation, reward, done, info

    def reset_model(self):
        self.action_phase_offset = 0
        self.initial_phase_offset = np.random.randint(0, 50) / 50

        qpos = self.gait_ref(self.phase)
        self.set_state(qpos, self.init_qvel)

        self.total_reward = 0
        self.force = self.get_force()

        self.midstance_impact_time = -1
        self.midstance_flag = False
        self.midstance_flag2 = False
        self.midstance_impact_applied = False
        self.record_once_flag = False
        self.post_impact_left_ankle =None
        self.post_impact_right_ankle =None



        observation = self._get_obs()
        return observation

    def reset_time_limit(self):
        if self.sim.data.time > self.max_ep_time:
            return self.reset()
        else:
            return self._get_obs()

    def get_total_reward(self):
        return self.total_reward
