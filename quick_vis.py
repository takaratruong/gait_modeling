
import random
import numpy as np
import torch.utils.data
import wandb
from utils.config_loader import load_args
from utils.amp_ppo import RL
from environments.walker2d.walker2d_env import WalkerEnv
from environments.humanoid.human_env_test2 import Humanoid_test_env2
from environments.humanoid.human_env_test import Humanoid_test_env

import time

from utils.video_callback import AMPVideoCallback
from utils.amp_models import ActorCriticNet

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecVideoRecorder
from stable_baselines3.common.env_util import make_vec_env


if __name__ == '__main__':
    args = load_args()

    #env = Humanoid_test_env(args=args)
    env = Humanoid_test_env2(args=args)
    print("env_created")

    num_inputs = env.observation_space.shape[0]
    num_outputs = env.action_space.shape[0]
    model = ActorCriticNet(num_inputs, num_outputs, [128, 128])
    model.load_state_dict(torch.load("results/models/human/human_iter6000.pt"))
    model.cuda()

    env.reset()
    obs = env.observe()

    num_samples = 1e6
    buffer = []

    while len(buffer) < num_samples:
        state = env.reset()
        for i in range(100000):
            print(state)
            with torch.no_grad():
                act = model.sample_best_actions(torch.from_numpy(state).cuda().type(torch.cuda.FloatTensor)).cpu().numpy()
                #act = np.zeros(29)#.cuda().type()
            next_state, rew, done, _ = env.step(act)
            env.render()

            buffer.append((state, next_state))

            state = next_state

            if done:
                state = env.reset()

            if len(buffer) % 10000 == 0:
                state = env.reset()
                print(len(buffer))

            if len(buffer) >= num_samples:
                break
            time.sleep(.01)
    #temp = np.array(buffer)

    #np.save('expert_motion.npy', temp)  # .npy extension is added if not given
    #d = np.load('test3.npy')

    #print(temp.shape)