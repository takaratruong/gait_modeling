import numpy as np

from gym import utils
import my_mujoco_env

from scipy.interpolate import interp1d

import ipdb

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -20.0,
}


class PD_Walker2dEnv(my_mujoco_env.MujocoEnv, utils.EzPickle):

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
        ref = np.loadtxt('2d_walking.txt')
        self.gait_ref = interp1d(np.arange(0, 11) / 10, ref, axis=0)
        self.gait_vel_ref = interp1d(np.arange(0, 11) / 10, np.diff(np.concatenate((np.zeros((1, 9)), ref))/.1, axis=0), axis=0)

        self.gait_cycle_time = 1.0
        self.time_offset = np.random.uniform(0, self.gait_cycle_time)  # offsets gait cycle time

        self._phase = 0
        self._max_phase = 1/.002
        self._phase_offset = np.random.randint(0, self._max_phase)

        self.max_ep_time = 8
        self.total_reward = 0

        # impulse
        self.impulse_duration = .2
        self.impulse_delay = 5 + np.random.uniform(0, self.gait_cycle_time)  # time between impulses
        self.impulse_time_start = 0  # used to keep track of elapsed time since impulse
        self.force = self.args.f
        self.lower_impact_lim = 100

        my_mujoco_env.MujocoEnv.__init__(self, xml_file, 10)
        self.init_qvel[0] = 1
        self.init_qvel[1:] = 0

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

        observation = np.concatenate((position, velocity/10, np.array([abs(self._phase % self._max_phase)/self._max_phase]))).ravel()

        return observation

    def observe(self):
        return self._get_obs()

    def get_total_reward(self):
        return self.total_reward

    def step(self, action):
        action = np.array(action.tolist())
        joint_action = action[0:6]
        phase_action = int(10*action[6]) #try with 100

        self._phase += phase_action  # allow agent to move phase forward/backward in time.
        joint_action =  action[0:6]
        phase_action = (self.get_phase() + self.frame_skip * 0.002) % 1

        ref = self.get_pos_ref(self._phase)  # get reference motion

        joint_target = joint_action + ref[3:]  # add reference motion to action for joints

        force = self.force if self.args.c_imp else self.get_rand_force()

        for _ in range(self.frame_skip):

            joint_obs = self.sim.data.qpos[3:]

            error = joint_target - joint_obs
            error_der = self.sim.data.qvel[3:]

            torque = self.p_gain*error - self.d_gain*error_der

            self.do_simulation(torque/100, 1)
            self._phase += 1

            # Apply force for specified duration
            if self.args.imp or self.args.c_imp:
                if (self.data.time - self.impulse_time_start) > self.impulse_delay:
                    self.sim.data.qfrc_applied[0] = force
                if (self.data.time - self.impulse_time_start - self.impulse_delay) >= self.impulse_duration:
                    self.sim.data.qfrc_applied[0] = 0
                    self.impulse_time_start = self.data.time

            # visualize reference motion
            if self.args.vis_ref:
                self.set_state(ref, self.gait_vel_ref(self.get_phase()))

        observation = self._get_obs()

        # Calculate Reward
        ref = self.get_pos_ref(self._phase)

        joint_ref = ref[3:]
        joint_obs = observation[2:8]
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
        self.total_reward += reward

        done = self.done
        info = {}
        if self.sim.data.time >= self.max_ep_time:
            info["TimeLimit.truncated"] = True
        else:
            info["TimeLimit.truncated"] = False
        # print(orient_reward, joint_reward, pos_reward)
        return observation, reward, done, info

    def get_pos_ref(self, phase):
        gait_ref = self.gait_ref(abs(phase % self._max_phase)/self._max_phase)

        gait_ref[0] += np.floor(phase/self._max_phase) - self.gait_ref((self._phase_offset % self._max_phase / self._max_phase))[0]  # add x-dist for every gait cycle so far and shift reference to match x init
        gait_ref[1] += 1.25  # match y of mujoco walker, walks underground otherwise..
        return gait_ref

    def get_vel_ref(self, phase):
        return self.gait_vel_ref(abs(phase % self._max_phase)/self._max_phase)

    #remove in next push
    def get_phase(self):
        return ((self.data.time + self.time_offset) % self.gait_cycle_time) / self.gait_cycle_time

    def get_rand_force(self):
        neg_force = np.random.randint(-abs(self.force), -self.lower_impact_lim)
        pos_force = np.random.randint(self.lower_impact_lim, abs(self.force))
        return neg_force if np.random.rand() <= .5 else pos_force

    def reset_model(self):
        # randomize starting phase alignment
        #self.time_offset = np.random.uniform(0, self.gait_cycle_time)

        self._phase_offset = np.random.randint(0, self._max_phase)
        self._phase = self._phase_offset

        init_pos_ref = self.get_pos_ref(self._phase)
        init_vel_ref = self.get_vel_ref(self._phase)

        self.impulse_delay = 5 + np.random.uniform(0, self.gait_cycle_time)
        self.impulse_time_start = self.data.time + .01

        #noise_low = -self._reset_noise_scale
        #noise_high = self._reset_noise_scale

        qpos = init_pos_ref  # self.init_qpos + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nq
        qvel = init_vel_ref  # self.init_qvel + self.np_random.uniform( low=noise_low, high=noise_high, size=self.model.nv)

        self.set_state(qpos, self.init_qvel)
        self.total_reward = 0

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)
