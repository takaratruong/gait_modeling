from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecVideoRecorder
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO

from environments.walker2d.walker2d_env import WalkerEnv
from environments.walker2d.walker2d_fixed_env import WalkerFixedEnv
from environments.humanoid.humanoid_env import HumanoidEnv
from environments.rajagopal.rajagopal_env import RajagopalEnv
from environments.skeleton.skeleton_env import SkeletonEnv


def load_envs(args, run):
    train_env, eval_env, vid_env = None, None, None

    if args.environment == 'walker2d':
        train_env = make_vec_env(lambda: WalkerEnv(args=args), n_envs=args.num_envs, seed=0, vec_env_cls=SubprocVecEnv)
        eval_env = make_vec_env(lambda: WalkerEnv(args=args), n_envs=1, vec_env_cls=SubprocVecEnv)

        if run is not None:
            vid_env = VecVideoRecorder(make_vec_env(lambda: WalkerEnv(args=args), n_envs=1, vec_env_cls=DummyVecEnv), args.log_dir + f"videos/{run.id}", record_video_trigger=lambda x: x == 0)

    if args.environment == 'walker2d_fixed':
        args.fixed_gait_policy = PPO.load(args.fixed_gait_policy_path) # <- path to model from gait_modeling directory'

        train_env = make_vec_env(lambda: WalkerFixedEnv(args=args), n_envs=args.num_envs, seed=0, vec_env_cls=SubprocVecEnv)
        eval_env = make_vec_env(lambda: WalkerFixedEnv(args=args), n_envs=1)

        if run is not None:
            vid_env = VecVideoRecorder(make_vec_env(lambda: WalkerFixedEnv(args=args), n_envs=1), args.log_dir + f"videos/{run.id}", record_video_trigger=lambda x: x == 0)

    if args.environment == 'humanoid':
        train_env = make_vec_env(lambda: HumanoidEnv(args=args), n_envs=args.num_envs, seed=0, vec_env_cls=SubprocVecEnv)
        eval_env = make_vec_env(lambda: HumanoidEnv(args=args), n_envs=1)

        if run is not None:
            vid_env = VecVideoRecorder(make_vec_env(lambda: HumanoidEnv(args=args), n_envs=1), args.log_dir + f"videos/{run.id}", record_video_trigger=lambda x: x == 0)

    if args.environment == 'rajagopal':
        train_env = make_vec_env(lambda: RajagopalEnv(args=args), n_envs=args.num_envs, seed=0, vec_env_cls=SubprocVecEnv)
        eval_env = make_vec_env(lambda: RajagopalEnv(args=args), n_envs=1)

        if run is not None:
            vid_env = VecVideoRecorder(make_vec_env(lambda: RajagopalEnv(args=args), n_envs=1), args.log_dir + f"videos/{run.id}", record_video_trigger=lambda x: x == 0)

    if args.environment == 'skeleton':
        train_env = make_vec_env(lambda: SkeletonEnv(args=args), n_envs=args.num_envs, seed=0, vec_env_cls=SubprocVecEnv)
        eval_env = make_vec_env(lambda: SkeletonEnv(args=args), n_envs=1)

        if run is not None:
            vid_env = VecVideoRecorder(make_vec_env(lambda: SkeletonEnv(args=args), n_envs=1), args.log_dir + f"videos/{run.id}", record_video_trigger=lambda x: x == 0)

    return train_env, eval_env, vid_env

