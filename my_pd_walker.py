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
    """
    ### Description
    This environment builds on the hopper environment based on the work done by Erez, Tassa, and Todorov
    in ["Infinite Horizon Model Predictive Control for Nonlinear Periodic Tasks"](http://www.roboticsproceedings.org/rss07/p10.pdf)
    by adding another set of legs making it possible for the robot to walker forward instead of
    hop. Like other Mujoco environments, this environment aims to increase the number of independent state
    and control variables as compared to the classic control environments. The walker is a
    two-dimensional two-legged figure that consist of four main body parts - a single torso at the top
    (with the two legs splitting after the torso), two thighs in the middle below the torso, two legs
    in the bottom below the thighs, and two feet attached to the legs on which the entire body rests.
    The goal is to make coordinate both sets of feet, legs, and thighs to move in the forward (right)
    direction by applying torques on the six hinges connecting the six body parts.
    ### Action Space
    The action space is a `Box(-1, 1, (6,), float32)`. An action represents the torques applied at the hinge joints.
    | Num | Action                                 | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
    |-----|----------------------------------------|-------------|-------------|----------------------------------|-------|--------------|
    | 0   | Torque applied on the thigh rotor      | -1          | 1           | thigh_joint                      | hinge | torque (N m) |
    | 1   | Torque applied on the leg rotor        | -1          | 1           | leg_joint                        | hinge | torque (N m) |
    | 2   | Torque applied on the foot rotor       | -1          | 1           | foot_joint                       | hinge | torque (N m) |
    | 3   | Torque applied on the left thigh rotor | -1          | 1           | thigh_left_joint                 | hinge | torque (N m) |
    | 4   | Torque applied on the left leg rotor   | -1          | 1           | leg_left_joint                   | hinge | torque (N m) |
    | 5   | Torque applied on the left foot rotor  | -1          | 1           | foot_left_joint                  | hinge | torque (N m) |
    ### Observation Space
    Observations consist of positional values of different body parts of the walker,
    followed by the velocities of those individual parts (their derivatives) with all the positions ordered before all the velocities.
    By default, observations do not include the x-coordinate of the top. It may
    be included by passing `exclude_current_positions_from_observation=False` during construction.
    In that case, the observation space will have 18 dimensions where the first dimension
    represent the x-coordinates of the top of the walker.
    Regardless of whether `exclude_current_positions_from_observation` was set to true or false, the x-coordinate
    of the top will be returned in `info` with key `"x_position"`.
    By default, observation is a `ndarray` with shape `(17,)` where the elements correspond to the following:
    | Num | Observation                                                       | Min                | Max                | Name (in corresponding XML file) | Joint | Unit |
    |-----|--------------------------------------------------------|----------------|-----------------|----------------------------------------|-------|------|
    | 0   | z-coordinate of the top (height of hopper)         | -Inf                 | Inf                | rootz (torso)      | slide | position (m) |
    | 1   | angle of the top                                   | -Inf                 | Inf                | rooty (torso)      | hinge | angle (rad) |
    | 2   | angle of the thigh joint                           | -Inf                 | Inf                | thigh_joint        | hinge | angle (rad) |
    | 3   | angle of the leg joint                             | -Inf                 | Inf                | leg_joint            | hinge | angle (rad) |
    | 4   | angle of the foot joint                            | -Inf                 | Inf                | foot_joint          | hinge | angle (rad) |
    | 5   | angle of the left thigh joint                      | -Inf                 | Inf                | thigh_left_joint  | hinge | angle (rad) |
    | 6   | angle of the left leg joint                        | -Inf                 | Inf                | leg_left_joint    | hinge | angle (rad) |
    | 7   | angle of the left foot joint                       | -Inf                 | Inf                | foot_left_joint  | hinge | angle (rad) |
    | 8   | velocity of the x-coordinate of the top            | -Inf                 | Inf                | rootx               | slide | velocity (m/s) |
    | 9   | velocity of the z-coordinate (height) of the top   | -Inf                 | Inf                | rootz                | slide | velocity (m/s)  |
    | 10  | angular velocity of the angle of the top           | -Inf                 | Inf                | rooty                | hinge | angular velocity (rad/s) |
    | 11  | angular velocity of the thigh hinge                | -Inf                 | Inf                | thigh_joint        | hinge | angular velocity (rad/s) |
    | 12  | angular velocity of the leg hinge                  | -Inf                 | Inf                | leg_joint            |  hinge | angular velocity (rad/s) |
    | 13  | angular velocity of the foot hinge                 | -Inf                 | Inf                | foot_joint           | hinge | angular velocity (rad/s) |
    | 14  | angular velocity of the thigh hinge                | -Inf                 | Inf                | thigh_left_joint   | hinge | angular velocity (rad/s) |
    | 15  | angular velocity of the leg hinge                  | -Inf                 | Inf                | leg_left_joint     | hinge | angular velocity (rad/s) |
    | 16  | angular velocity of the foot hinge                 | -Inf                 | Inf                | foot_left_joint    | hinge | angular velocity (rad/s) |
    ### Rewards
    The reward consists of three parts:
    - *healthy_reward*: Every timestep that the walker is alive, it receives a fixed reward of value `healthy_reward`,
    - *forward_reward*: A reward of walking forward which is measured as
    *`forward_reward_weight` * (x-coordinate before action - x-coordinate after action)/dt*.
    *dt* is the time between actions and is dependeent on the frame_skip parameter
    (default is 4), where the frametime is 0.002 - making the default
    *dt = 4 * 0.002 = 0.008*. This reward would be positive if the walker walks forward (right) desired.
    - *ctrl_cost*: A cost for penalising the walker if it
    takes actions that are too large. It is measured as
    *`ctrl_cost_weight` * sum(action<sup>2</sup>)* where *`ctrl_cost_weight`* is
    a parameter set for the control and has a default value of 0.001
    The total reward returned is ***reward*** *=* *healthy_reward bonus + forward_reward - ctrl_cost* and `info` will also contain the individual reward terms
    ### Starting State
    All observations start in state
    (0.0, 1.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    with a uniform noise in the range of [-`reset_noise_scale`, `reset_noise_scale`] added to the values for stochasticity.
    ### Episode Termination
    The walker is said to be unhealthy if any of the following happens:
    1. Any of the state space values is no longer finite
    2. The height of the walker is ***not*** in the closed interval specified by `healthy_z_range`
    3. The absolute value of the angle (`observation[1]` if `exclude_current_positions_from_observation=False`, else `observation[2]`) is ***not*** in the closed interval specified by `healthy_angle_range`
    If `terminate_when_unhealthy=True` is passed during construction (which is the default),
    the episode terminates when any of the following happens:
    1. The episode duration reaches a 1000 timesteps
    2. The walker is unhealthy
    If `terminate_when_unhealthy=False` is passed, the episode is terminated only when 1000 timesteps are exceeded.
    ### Arguments
    No additional arguments are currently supported in v2 and lower.
    ```
    env = gym.make('Walker2d-v2')
    ```
    v3 and beyond take gym.make kwargs such as xml_file, ctrl_cost_weight, reset_noise_scale etc.
    ```
    env = gym.make('Walker2d-v3', ctrl_cost_weight=0.1, ....)
    ```
    | Parameter               | Type       | Default      |Description                    |
    |-------------------------|------------|--------------|-------------------------------|
    | `xml_file`              | **str**    | `"walker2d.xml"` | Path to a MuJoCo model |
    | `forward_reward_weight` | **float**  | `1.0`        | Weight for *forward_reward* term (see section on reward) |
    | `ctrl_cost_weight`      | **float**  | `1e-3`       | Weight for *ctr_cost* term (see section on reward) |
    | `healthy_reward`        | **float**  | `1.0`        | Constant reward given if the ant is "healthy" after timestep |
    | `terminate_when_unhealthy` | **bool**| `True`       | If true, issue a done signal if the z-coordinate of the walker is no longer healthy |
    | `healthy_z_range`       | **tuple**  | `(0.8, 2)`   | The z-coordinate of the top of the walker must be in this range to be considered healthy |
    | `healthy_angle_range`   | **tuple**  | `(-1, 1)`    | The angle must be in this range to be considered healthy|
    | `reset_noise_scale`     | **float**  | `5e-3`       | Scale of random perturbations of initial position and velocity (see section on Starting State) |
    | `exclude_current_positions_from_observation`| **bool** | `True`| Whether or not to omit the x-coordinate from observations. Excluding the position can serve as an inductive bias to induce position-agnostic behavior in policies |
    ### Version History
    * v3: support for gym.make kwargs such as xml_file, ctrl_cost_weight, reset_noise_scale etc. rgb rendering comes from tracking camera (so agent does not run away from screen)
    * v2: All continuous control environments now use mujoco_py >= 1.50
    * v1: max_time_steps raised to 1000 for robot based tasks. Added reward_threshold to environments.
    * v0: Initial versions release (1.0.0)
    """

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

        self.max_ep_time = 30
        self.total_reward = 0

        # impulse
        self.impulse_duration = .2
        self.impulse_delay = 5 + np.random.uniform(0, self.gait_cycle_time)  # time between impulses
        self.impulse_time_start = 0  # used to keep track of elapsed time since impulse
        self.force = self.args.f
        self.lower_impact_lim = 100

        my_mujoco_env.MujocoEnv.__init__(self, xml_file, 10)
        self.init_qvel[0] = 1

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

        observation = np.concatenate((position, velocity / 10, np.array([self.get_phase()]))).ravel()

        return observation

    def observe(self):
        return self._get_obs()

    def get_total_reward(self):
        return self.total_reward

    def step(self, action):
        action = np.array(action.tolist())
        joint_action =  action[0:6]
        phase_action = (self.get_phase() + 0.2) % 1

        ref = self.get_gait_ref(phase_action)  # get reference motion
        joint_target = joint_action + ref[3:]  # add reference motion to action for joints

        force = self.force if self.args.c_imp else self.get_rand_force()

        for _ in range(self.frame_skip):

            joint_obs = self.sim.data.qpos[3:]

            error = joint_target - joint_obs
            error_der = self.sim.data.qvel[3:]

            torque = self.p_gain*error - self.d_gain*error_der

            self.do_simulation(torque/100, 1)

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
        # phase = self.get_phase()
        # ref = self.get_gait_ref(phase)

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
        # if orient_reward < 0.6 or joint_reward < 0.6 or pos_reward < 0.6:
        #     done = True
        info = {}
        # print(orient_reward, joint_reward, pos_reward)
        return observation, reward, done, info

    def get_gait_ref(self, phase):
        gait_ref = self.gait_ref(phase)
        gait_ref[0] += np.floor((self.data.time + self.time_offset) / self.gait_cycle_time) - \
                       self.gait_ref((self.time_offset % self.gait_cycle_time) / self.gait_cycle_time)[
                           0]  # add x-dist for every gait cycle so far and shift reference to match x init
        gait_ref[1] += 1.25  # match y of mujoco walker, walks underground otherwise..
        return gait_ref

    def get_phase(self):
        return ((self.data.time + self.time_offset) % self.gait_cycle_time) / self.gait_cycle_time

    def get_rand_force(self):
        neg_force = np.random.randint(-abs(self.force), -self.lower_impact_lim)
        pos_force = np.random.randint(self.lower_impact_lim, abs(self.force))
        return neg_force if np.random.rand() <= .5 else pos_force

    def reset_model(self):
        # randomize starting phase alignment
        self.time_offset = np.random.randint(0, 50) / 50.0 #np.random.uniform(0, self.gait_cycle_time)
        init_pos_ref = self.get_gait_ref(self.get_phase())
        init_vel_ref = self.gait_vel_ref(self.get_phase())

        self.impulse_delay = 5 + np.random.uniform(0, self.gait_cycle_time)
        self.impulse_time_start = self.data.time + .001  # s

        #noise_low = -self._reset_noise_scale
        #noise_high = self._reset_noise_scale

        qpos = init_pos_ref  # self.init_qpos + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nq
        qvel = init_vel_ref  # self.init_qvel + self.np_random.uniform( low=noise_low, high=noise_high, size=self.model.nv)

        self.set_state(qpos, qvel)
        self.total_reward = 0

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)
