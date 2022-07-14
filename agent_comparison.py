import random
import numpy as np
import torch.utils.data
import wandb
from utils.config_loader import load_args
from utils.amp_ppo import RL
from environments.walker2d.walker2d_env import WalkerEnv
from environments.humanoid.human_env_test2 import Humanoid_test_env2
from environments.humanoid.human_env_test import Humanoid_test_env
import ipdb
import pandas as pd
import time

from utils.video_callback import AMPVideoCallback
from utils.amp_models import ActorCriticNet

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecVideoRecorder
from stable_baselines3.common.env_util import make_vec_env



def midstance_response(env, model, force, axis, render):
    env.args.perturbation_force = force
    env.args.perturbation_dir = axis

    env.reset()
    env.args.perturbation_force = force
    env.args.perturbation_dir = axis

    state = env.observe()

    left_ankle = None
    right_ankle = None
    success = True

    done = False
    while not done:
        with torch.no_grad():
            act = model.sample_best_actions(
                torch.from_numpy(state).cuda().type(torch.cuda.FloatTensor)).cpu().numpy()
        next_state, _, done, info = env.step(act)

        if render:
            env.render()

        if done:
            left_ankle = info['left_ankle']
            right_ankle = info['right_ankle']
            success = info['is_healthy']

        state = next_state

    results = np.zeros(5)
    results[0:2] = left_ankle
    results[2:4] = right_ankle
    results[4] = success
    return results


if __name__ == '__main__':
    args = load_args()

    walk_env = Humanoid_test_env2(args=args)
    pert_env = Humanoid_test_env2(args=args)
    amp_env  = Humanoid_test_env2(args=args)

    num_inputs = walk_env.observation_space.shape[0]
    num_outputs = walk_env.action_space.shape[0]

    walk_model = ActorCriticNet(num_inputs, num_outputs, [128, 128])
    walk_model.load_state_dict(torch.load("results/models/human_4/human_4_iter400.pt"))
    walk_model.cuda()

    pert_model = ActorCriticNet(num_inputs, num_outputs, [128, 128])
    pert_model.load_state_dict(torch.load("results/models/human_pert/human_pert_iter2200.pt"))
    pert_model.cuda()

    compiled_results = np.zeros(5)
    env = pert_env
    model = pert_model
    num_trials = 5
    force = -200
    axis = 1

    for _ in range(num_trials):
        result = midstance_response(env, model, force, axis, True)
        #print(result)
        compiled_results = np.vstack((compiled_results, result))

    #print(compiled_results.shape)
    #print(compiled_results)

    #np.savetxt('pert50/pert_forward_50.txt', compiled_results)
