import torch.nn as nn
import torch.nn.functional as F
import torch
from stable_baselines3.common.callbacks import BaseCallback, sync_envs_normalization
import numpy as np
import random

class Discriminator(nn.Module):
    def __init__(self, num_inputs, hidden_layer=(64, 64)):
        super().__init__()

        self.l1 = nn.Linear(num_inputs, hidden_layer[0])
        self.l2 = nn.Linear(hidden_layer[0], hidden_layer[1])
        self.l3 = nn.Linear(hidden_layer[1], 1)

    def forward(self, state, next_state):
        x = torch.cat([state, next_state], dim=-1)

        out = F.relu(self.l1(x))
        out = F.relu(self.l2(out))
        out = F.relu(self.l3(out))
        return out

    def reward(self, state, next_state):
        with torch.no_grad():
            out = self.forward(state, next_state)
            reward = torch.clamp(1 - (1.0/4) * torch.square(out - 1), min=0)
        return reward.squeeze()


class UpdateDiscriminator(BaseCallback):
    def __init__(self, env, batch_size=200, verbose=0):
        super(UpdateDiscriminator, self).__init__(verbose)
        self.batch_size = batch_size
        self.env = env

    def _on_step(self):
        pass

    def sample_agent(self):
        # rollout episodes then sample from it.
        rollout = []
        while len(rollout) < self.batch_size*2:
            state = self.env.reset()

            done = False
            while not done:
                action, _ = self.model.predict(state, deterministic=False)
                next_state, _, done, _ = self.env.step(action)

                rollout.append((state, next_state))
                if len(rollout) >= self.batch_size*2:
                    break

                state = next_state
        self.env.close()

        rollout = np.array(rollout)
        indices = np.random.choice(len(rollout), self.batch_size)

        agent_batch = rollout[indices]
        return agent_batch

    def sample_prior(self):


    def _on_rollout_end(self):
        sample_agent.





