import torch.nn as nn
import torch.nn.functional as F
import torch
from stable_baselines3.common.callbacks import BaseCallback, sync_envs_normalization


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
    def __init__(self, batch_size=100, verbose=0):
        super(UpdateDiscriminator, self).__init__(verbose)
        self.batch_size = batch_size

    def _on_step(self):
        pass

    def _on_rollout_end(self):

        rollout = self.model.rollout_buffer.get(self.batch_size)

        for i, x in enumerate(rollout):
            print(i)
            print(len(x[0]))
            print(x[0])
            print(len(x[1]))
            print(x[1])
            print()

        print()
        print(z)

"""

class UpdateDiscriminator(BaseCallback):
    def __init__(self, env):
        super(UpdateDiscriminator, self).__init__()
        self.env = env
        self.batch_size = 100

    def _on_rollout_end(self):
        pass
        #print(self.rollout_buffer)
        #print(self.model.rollout_buffer.get(self.batch_size))
"""