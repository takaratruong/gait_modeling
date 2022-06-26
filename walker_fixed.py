import numpy as np
from walker_base import WalkerBase
import ipdb


class WalkerFixed(WalkerBase):
    def step(self, action):
        new_action = np.array(action.tolist())

        fixed_gait_action, _ = self.args.fixed_gait_policy.predict(self._get_obs())
        fixed_gait_action = np.array(fixed_gait_action.tolist())

        action = new_action + fixed_gait_action

        if self.phase < 0.5:
            joint_action = action[0:6].copy()
        else:
            joint_action = action[[3, 4, 5, 0, 1, 2]].copy()

        self.sim_cntr += int(self.args.phase_action_mag * action[6])

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

        done = self.done
        info = {}

        self.elapsed_steps+=1
        if self.elapsed_steps >= self.max_ep_steps:
            info["TimeLimit.truncated"] = not done
            done = True

        return observation, reward, done, info

