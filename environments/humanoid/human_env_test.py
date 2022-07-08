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
PARAMS_KP_KD = {"chest": [1000, 100], "neck": [100, 10], "right_shoulder": [400, 40], "right_elbow": [300, 30],
        "left_shoulder": [400, 40], "left_elbow": [300, 30], "right_hip": [500, 50], "right_knee": [500, 50],
        "right_ankle": [400, 40], "left_hip": [500, 50], "left_knee": [500, 50], "left_ankle": [400, 40]}
KP_GAIN = np.array(
         [1000, 1000, 1000,  # chest
          100, 100, 100,  # neck
          400, 400, 400,  # shoulder
          300,  # elbow
          400, 400, 400,  # shoulder
          300,  #elbow
          500, 500, 500,  #hip
          500,  # knee
          400, 400, 400, # ankle
          500, 500, 500,  # hip
          500,  # knee
          400, 400, 400 # ankle
          ])
KD_GAIN = np.array(
         [100, 100, 100,  # chest
          10, 10, 10,  # neck
          40, 40, 40,  # shoulder
          30,  # elbow
          40, 40, 40,  # shoulder
          30,  #elbow
          50, 50, 50,  #hip
          50,  # knee
          40, 40, 40, # ankle
          50, 50, 50,  # hip
          50,  # knee
          40, 40, 40 # ankle
          ])

class Humanoid_test_env(mujoco_env_humanoid.MujocoEnv, utils.EzPickle):

    def __init__(
            self,
            xml_file="humanoid.xml",
            forward_reward_weight=1.0,
            ctrl_cost_weight=1e-3,
            healthy_reward=1.0,
            terminate_when_unhealthy=True,
            healthy_z_range=(0.7, 10.0),
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

        self.gait_cycle_time = 38 * 0.0333320000

        self.gait_ref = interp1d(np.arange(0, 39) / 38, ref, axis=0)
        #self.gait_vel_ref = interp1d(np.arange(0, 11) / 10, np.diff(np.concatenate((np.zeros((1, 9)), ref))/.1, axis=0), axis=0)

        self._max_phase = 38 * 0.0333320000/.016 # 500 # 1/.002

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

            observation = np.concatenate((flipped_pos, flipped_vel,  np.array([self.phase - 0.5])))

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
        gait_ref[0] += 1.23859 * np.floor(( self.sim_cntr + cntr) / self._max_phase) #- self.gait_ref((self.offset % self._max_phase)/self._max_phase)[0]
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

        #self.sim_cntr += int(self.args.phase_action_mag * action[28])


        """"""
        phase_temp= (self.data.time % self.gait_cycle_time)/self.gait_cycle_time
        pos_temp = self.gait_ref(phase_temp)
        pos_temp[0] += 1.23859 * np.floor(self.data.time / self.gait_cycle_time)
        """"""""

        ref = self.cntr2ref(self.sim_cntr + self.frame_skip)

        joint_target = joint_action + ref[7:]  # add action to joint ref to create final joint target

        for _ in range(self.frame_skip):
            joint_obs = self.sim.data.qpos[7:]
            joint_vel_obs = self.sim.data.qvel[6:]
            print("----------")
            print(joint_obs)
            print(joint_target)

            error = joint_target - joint_obs

            print("error", error)
            error_der = joint_vel_obs

            #torque = KP_GAIN * error - KD_GAIN * error_der
            torque = 10000 * error - 10 * error_der
            self.do_simulation(torque, 1)

            self.sim_cntr += 1

            # Apply force for specified duration
            if self.args.const_perturbation or self.args.rand_perturbation:
                if (self.data.time - self.impulse_time_start) > self.impulse_delay:
                    self.sim.data.qfrc_applied[0] = self.force

                if (self.data.time - self.impulse_time_start - self.impulse_delay) >= self.impulse_duration:
                    self.sim.data.qfrc_applied[0] = 0
                    self.impulse_time_start = self.data.time
                    self.force = self.args.perturbation_force if self.args.const_perturbation else self.get_rand_force()

        self.set_state(ref, self.init_qvel)

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
        orient_reward = 2 * np.sum((orient_ref - orient_obs) ** 2) + 5 * np.sum((self.sim.data.qvel[3:6]) ** 2)
        orient_reward = np.exp(-orient_reward)

        reward = self.args.orient_weight * orient_reward + self.args.joint_weight * joint_reward + self.args.pos_weight * pos_reward

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
        self.offset = int(np.random.randint(0, self._max_phase / 10) * 10)
        self.sim_cntr = self.offset

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
