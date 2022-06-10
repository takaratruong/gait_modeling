from stable_baselines3.common.vec_env import VecVideoRecorder
import wandb
from stable_baselines3 import PPO
from walker_base import WalkerBase
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from config_loader import load_args
from utils import VideoCallback

policy_kwargs = dict(log_std_init=-2.0, ortho_init=True, activation_fn=torch.nn.ReLU,
                     net_arch=[dict(pi=[128, 128], vf=[128, 128])])


if __name__ == '__main__':
    args = load_args()

    run = wandb.init(project="gait-modeling", config=args, name=args.exp_name, sync_tensorboard=True,
                     monitor_gym=True, save_code=True, dir=args.log_dir)
    

    train_env = make_vec_env(lambda: WalkerBase(args=args), n_envs=args.num_cpu, vec_env_cls=SubprocVecEnv)

    eval_callback = EvalCallback(train_env, best_model_save_path=args.log_dir + '/models/' + args.exp_name,
                                 log_path=args.log_dir + '/models/' + args.exp_name,
                                 eval_freq=max(args.eval_freq * args.num_cpu, 1),
                                 deterministic=True, render=False)

    vid_env = VecVideoRecorder(make_vec_env(lambda: WalkerBase(args=args)), args.log_dir + f"videos/{run.id}",
                               record_video_trigger=lambda x: x == 0)
    
    vid_callback = VideoCallback(vid_env, eval_freq=args.vid_freq * args.num_cpu)
    

    # num_steps* num_env = const
    # 300 * 1 = 300
    # 300 * 20 /x = 300 

    #num_steps * num_env = const
    #300 * 20 = 6000
    #300 * 100 = 30000
    #300 * 100/5 = 6000
    
    model = PPO("MlpPolicy", train_env, batch_size=(args.num_cpu * args.num_steps) // 4, use_sde=False,
                n_steps=args.num_steps, n_epochs=args.num_epochs, gamma=.99, gae_lambda=.95, target_kl=.01,
                verbose=True, clip_range=.1, ent_coef=.000585045, vf_coef=0.871923, max_grad_norm=10,
                learning_rate=1e-4, tensorboard_log=args.log_dir + 'tf/', policy_kwargs=policy_kwargs,device='cpu')

    model.learn(total_timesteps=args.time_steps,  callback=[eval_callback, vid_callback], tb_log_name=args.exp_name)

    run.finish()
