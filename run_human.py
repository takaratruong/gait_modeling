import random
import numpy as np
import torch.utils.data
import wandb
from utils.config_loader import load_args
from utils.amp_ppo import RL
from environments.humanoid.human_env_test import Humanoid_test_env
from utils.video_callback import AMPVideoCallback

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecVideoRecorder
from stable_baselines3.common.env_util import make_vec_env

if __name__ == '__main__':
    # set seed
    seed = 2
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)

    args = load_args()
    run = wandb.init(project="amp", config=args, name=args.exp_name, monitor_gym=True, dir=args.log_dir)

    train_env = make_vec_env(lambda: Humanoid_test_env(args=args), n_envs=args.num_envs, seed=0, vec_env_cls=SubprocVecEnv)

    vid_env = VecVideoRecorder(make_vec_env(lambda: Humanoid_test_env(args=args), n_envs=1), args.log_dir + f"videos/{run.id}",
                               record_video_trigger=lambda x: x == 0)
    vid_callback = AMPVideoCallback(vid_env)

    ppo = RL(train_env, vid_callback, args, [128, 128])

    ppo.collect_samples_multithread()
