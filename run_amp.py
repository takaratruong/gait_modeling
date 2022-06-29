from stable_baselines3.common.vec_env import VecVideoRecorder
import wandb
from stable_baselines3 import PPO
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from config_loader import load_args
from utils import VideoCallback
from amp_models import UpdateDiscriminator
from amp_models import Discriminator
from amp_env import AmpEnv


policy_kwargs = dict(log_std_init=-2.0, ortho_init=True, activation_fn=torch.nn.ReLU,
                     net_arch=[dict(pi=[128, 128], vf=[128, 128])])


if __name__ == '__main__':
    args = load_args()
    args.discriminator = Discriminator(num_inputs=36)

    train_env = make_vec_env(lambda: AmpEnv(args=args), n_envs=1, seed=0, vec_env_cls=DummyVecEnv)
    eval_env = make_vec_env(lambda: AmpEnv(args=args), n_envs=1, vec_env_cls=DummyVecEnv)

    eval_callback = EvalCallback(eval_env, best_model_save_path=args.log_dir + args.exp_name, log_path=args.log_dir + 'models/' + args.exp_name, eval_freq=args.eval_freq * args.num_epochs * args.num_envs, deterministic=True, render=False)
    disc_callback = UpdateDiscriminator()

    model = PPO("MlpPolicy", train_env, batch_size=(args.num_envs*args.num_steps)//4, use_sde=False, n_steps= args.num_steps, n_epochs=args.num_epochs, gamma=.99, gae_lambda=.95,
                target_kl=.01, verbose=True,  clip_range=.1, ent_coef=.000585045, vf_coef=0.871923, max_grad_norm=10, learning_rate=1e-4,
                tensorboard_log = args.log_dir + 'tf/',policy_kwargs=policy_kwargs)
    model.learn(total_timesteps=args.time_steps, callback=[eval_callback, disc_callback], tb_log_name=args.exp_name)

