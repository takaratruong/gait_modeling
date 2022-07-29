import random
import numpy as np
import torch.utils.data
import wandb
from utils.config_loader import load_args
from utils.amp_ppo import RL
from utils.amp_models import set_seed
from environments.humanoid.humanoid_env import HumanoidEnv
from environments.skeleton.skeleton_env import SkeletonEnv

from utils.video_callback import AMPVideoCallback
from utils.load_envs import load_envs

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecVideoRecorder
from stable_baselines3.common.env_util import make_vec_env

if __name__ == '__main__':
    args = load_args()
    seed = 2
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)

    print("Exp Name: ", args.exp_name)
    print("Config File:", args.config)
    print("Algorithm:", args.alg)
    run = None
    if args.wandb:
        run = wandb.init(project=args.project_name, config=args, name=args.exp_name, monitor_gym=True, dir=args.log_dir)

    train_env, _, vid_env = load_envs(args, run)

    vid_callback = AMPVideoCallback(vid_env) if vid_env is not None else None

    ppo = RL(train_env, vid_callback, args, [128, 128])

    ppo.collect_samples_multithread()





