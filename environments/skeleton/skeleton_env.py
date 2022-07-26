import numpy as np
from gym import utils
from scipy.interpolate import interp1d
import ipdb
import os
from scipy.spatial.transform import Rotation as R
import time

import environments.skeleton.skeleton_mujoco_env as mujoco_env_skeleton
from environments.skeleton.skeleton_utils import reflect_action, reflect_sagital


class SkeletonEnv(mujoco_env_skeleton.MujocoEnv, utils.EzPickle):

    def __init__(
            self,
            xml_file="skeleton_foot_contact_no_treadmill.xml",
            terminate_when_unhealthy=True,
            healthy_z_range=(.85, 10.0), #1.15
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
        # args.gait_ref_file = '../environments/skeleton/skeleton_walk_ref.txt'
        # ref = np.load('../environments/skeleton/skeleton_walk_ref.npy')
        ref = np.load(args.gait_ref_file)
        # print(ref.shape)
        self.gait_ref = interp1d(np.arange(0, ref.shape[0]) / (ref.shape[0]-1), ref, axis=0)

        self.treadmill_velocity = args.treadmill_velocity
        self.gait_cycle_time = args.gait_cycle_time # 38 * 0.03333200
        self.time_step = .01

        self.initial_phase_offset = np.random.randint(0, 50) / 50

        self.action_phase_offset = 0

        self.max_ep_time = self.args.max_ep_time
        self.frame_skip = self.args.frame_skip

        self.total_reward = 0

        self.force = self.get_force()

        mujoco_env_skeleton.MujocoEnv.__init__(self, xml_file, self.frame_skip)
        self.init_qvel[0] = 1
        qpos = self.gait_ref(self.initial_phase_offset)
        qpos[1] -= .3
        self.set_state(qpos[:-1], self.init_qvel)

    @property
    def is_healthy(self):
        min_z, max_z = self._healthy_z_range
        is_healthy = min_z < self.sim.data.get_body_xpos('torso')[2] < max_z
        # is_healthy = True
        # print(self.sim.data.get_body_xpos('torso'))

        return is_healthy

    def calc_reward(self, ref):
        # joint_names = ['hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 'walker_knee_r', 'ankle_angle_r', 'subtalar_angle_r', 'mtp_angle_r', 'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l', 'walker_knee_l', 'ankle_angle_l', 'subtalar_angle_l', 'mtp_angle_l', 'lumbar_extension', 'lumbar_bending', 'lumbar_rotation', 'arm_flex_r', 'arm_add_r', 'arm_rot_r', 'elbow_flex_r', 'arm_flex_l', 'arm_add_l', 'arm_rot_l', 'elbow_flex_l']

        # jnt2addr = self.model.get_joint_qpos_addr

        ##joint reward
        # joint_err = np.sum([(self.sim.data.qpos[jnt2addr(joint)] - ref[jnt2addr(joint)]) ** 2 for joint in joint_names])
        # joint_err = np.sum( (self.sim.data.qpos[6:-1] - ref[6:-1] ) ** 2 )
        joint_err = np.sum( (self.sim.data.qpos[6:] - ref[6:-1] ) ** 2 )

        joint_rew = np.exp(-2 * joint_err)

        # root pos/vel reward, Note: world frame z-up , model frame y-up
        pos_err = 10 * (1.25 - self.sim.data.get_body_xvelp('torso')[0]) ** 2 + \
                  1 * (0 - self.sim.data.get_body_xvelp('torso')[1]) ** 2 + \
                  1 * (0 - self.sim.data.get_body_xpos('torso')[1]) ** 2
                # 3 * (0 - self.sim.data.get_body_xpos('torso')[0]) ** 2 + \

        # print(self.sim.data.get_body_xvelp('torso'))
        pos_rew = np.exp(-1 * pos_err)

        # rotation reward
        rot_err = 5 * np.sum(self.sim.data.get_body_xvelr('torso') ** 2)
        rot_rew = np.exp(-1 * rot_err)

        reward = self.args.rot_weight * rot_rew + self.args.jnt_weight * joint_rew + self.args.pos_weight * pos_rew

        return reward

    @property
    def done(self):
        done = not self.is_healthy if self._terminate_when_unhealthy else False
        return done

    def _get_obs(self):

        if self.phase <= .5:
            position = self.sim.data.qpos.flat.copy()
            # position = position[1:-1]  # exclude treadmill and x
            position = position[1:]  # exclude treadmill and x

            velocity = np.clip(self.sim.data.qvel.flat.copy(), -1000, 1000)
            #velocity = velocity[:-1]
            observation = np.concatenate((position, velocity / 10, np.array([self.phase]))).ravel()
        else:
            flipped_pos = reflect_sagital(self.sim.data.qpos.flat.copy(), self.model.get_joint_qpos_addr)
            # flipped_pos = flipped_pos[1:-1]
            flipped_pos = flipped_pos[1:]

            flipped_vel = reflect_sagital( np.clip(self.sim.data.qvel.flat.copy(), -1000, 1000), self.model.get_joint_qvel_addr)
            #flipped_vel = flipped_vel[:-1]

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
    def target_reference(self):         # Same shape as qpos
        frame_skip_time = self.frame_skip * self.time_step
        initial_offset_time = self.initial_phase_offset * self.gait_cycle_time
        action_offset_time = self.action_phase_offset * self.gait_cycle_time
        total_time = frame_skip_time + initial_offset_time + action_offset_time + self.data.time

        phase_target = (total_time % self.gait_cycle_time) / self.gait_cycle_time

        if phase_target <= .5:
            gait_ref = self.gait_ref(phase_target)
        else:
            # only flipped because reference motion of full gait cycle is not symmetrical.
           gait_ref = self.gait_ref(phase_target - .5)
           gait_ref = reflect_sagital(gait_ref, self.model.get_joint_qpos_addr)

        gait_ref[1] -= .3
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
        action = np.array(action.tolist()).copy()
        # ipdb.set_trace()
        joint_action = action[:-1] if self.phase <= .5 else reflect_action(action[:-1], self.model.actuator_name2id)

        phase_action = self.args.phase_action_mag * action[-1]

        self.action_phase_offset += phase_action

        target_ref = self.target_reference  # change later

        # Treadmill
        # print(len(joint_action))
        final_target = joint_action + target_ref[6:-1]  # add action to joint ref to create final joint target

        for _ in range(self.frame_skip):
            # Treadmill
            joint_obs = self.sim.data.qpos[6:].copy()
            joint_vel_obs = self.sim.data.qvel[6:].copy()

            error = final_target - joint_obs
            error_der = joint_vel_obs

            torque = 120 * error - 10 * error_der

            # Treadmill
            # self.sim.data.set_joint_qvel('treadmill', 0)#-self.treadmill_velocity)  # keep treadmill moving

            self.do_simulation(torque / 100, 1)

            # self.set_state(self.target_reference[:-1], self.init_qvel)

        observation = self._get_obs()

        reward = self.calc_reward(target_ref)

        # print(reward)
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

        #Treadmill
        qpos[1] -= .3

        self.set_state(qpos[:-1], self.init_qvel)

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