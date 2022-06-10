import numpy as np
from gym import utils
import my_mujoco_env
from scipy.interpolate import interp1d

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -20.0,
}

class WalkerBase(my_mujoco_env.MujocoEnv, utils.EzPickle):
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

        # PD Controller Values
        self.p_gain = 100
        self.d_gain = 10

        # Gait Reference
        ref = np.loadtxt('2d_walking.txt')
        self.gait_ref = interp1d(np.arange(0, 11) / 10, ref, axis=0)
        self.gait_vel_ref = interp1d(np.arange(0, 11) / 10, np.diff(np.concatenate((np.zeros((1, 9)), ref)) / .1, axis=0), axis=0)

        self._phase_cntr = 0
        self._max_phase = 500  # 1/.002

        self.max_ep_time = self.args.max_ep_time #10  # seconds
        self.total_reward = 0

        self.gait_cycle_time = 1.0
        self.time_offset = np.random.uniform(0, self.gait_cycle_time)  # offsets gait cycle time

        # impulse
        self.force = self.args.perturbation_force
        self.impulse_duration = self.args.perturbation_duration
        self.impulse_delay = self.args.perturbation_delay # 3.5
        self.impulse_time_start = 0

        self._max_episode_steps = self.args.num_steps #self.args.max_ep_steps
        self._elapsed_steps = 0


        my_mujoco_env.MujocoEnv.__init__(self, xml_file, self.args.frame_skip)
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

        #if self.sim.data.time > self.max_ep_time:
        #    is_healthy = False
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
        action = np.array(action.tolist())

        if self.cntr2phase(self._phase_cntr) < 0.5:
            joint_action = action[0:6].copy()
        else:
            joint_action = action[[3, 4, 5, 0, 1, 2]].copy()

        phase_action = int(self.args.phase_action_mag * action[6])

        self._phase_cntr += phase_action

        target_phase_cntr = self._phase_cntr + self.frame_skip

        target_ref = self.get_pos_ref(self.cntr2phase(target_phase_cntr))

        joint_target = joint_action * 0 + target_ref[3:]  # add action to joint ref to create final joint target

        for _ in range(self.frame_skip):

            joint_obs = self.sim.data.qpos[3:]
            joint_vel_obs = self.sim.data.qvel[3:]

            error = joint_target - joint_obs
            error_der = joint_vel_obs

            torque = self.p_gain * error - self.d_gain * error_der

            self.do_simulation(torque / 100, 1)
            self._phase_cntr += 1

            # Apply force for specified duration
            if self.args.const_perturbation or self.args.rand_perturbation:
                if (self.data.time - self.impulse_time_start) > self.impulse_delay:
                    self.sim.data.qfrc_applied[0] = self.force

                if (self.data.time - self.impulse_time_start - self.impulse_delay) >= self.impulse_duration:
                    self.sim.data.qfrc_applied[0] = 0
                    self.impulse_time_start = self.data.time
                    self.force = self.args.perturbation_force if self.args.const_perturbation else self.get_rand_force()

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

        reward = self.args.orient_weight * orient_reward + self.args.joint_weight * joint_reward + self.args.pos_weight * pos_reward

        self.total_reward += reward

        observation = self._get_obs()

        done = self.done
        info = {}
        #if self.sim.data.time >= self.max_ep_time:
        #    info["TimeLimit.truncated"] = True
        #else:
        #    info["TimeLimit.truncated"] = False
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info["TimeLimit.truncated"] = not done
            done = True

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
        neg_force = np.random.randint(-abs(self.args.perturbation_force), -abs(self.args.min_perturbation_force_mag))
        pos_force = np.random.randint(abs(self.args.min_perturbation_force_mag), abs(self.args.perturbation_force))
        return neg_force if np.random.rand() <= .5 else pos_force

    def reset_model(self):
        self._elapsed_steps = 0

        # randomize starting phase alignment
        offset = int(np.random.randint(0, self._max_phase / 10) * 10)

        self._phase_cntr = offset
        init_pos_ref = self.get_pos_ref(self.cntr2phase(self._phase_cntr))

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
