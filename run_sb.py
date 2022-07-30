import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from configs.config_loader import load_args, POLICY_KWARGS
from utils.load_envs import load_environments
from utils.video_callback import VideoCallback


if __name__ == '__main__':
    args = load_args()

    # Initialize wandb
    run = wandb.init(project=args.project_name, config=args, name=args.exp_name, sync_tensorboard=True, monitor_gym=True, save_code=True, dir=args.log_dir) if args.wandb else None

    # Load environments
    train_env, eval_env, vid_env = load_environments(args, run)

    # Create callbacks
    eval_callback = EvalCallback(eval_env, best_model_save_path=args.log_dir + 'models/' + args.exp_name, log_path=args.log_dir + 'models/' + args.exp_name, eval_freq=args.eval_freq * args.num_epochs * args.num_envs, deterministic=True, render=False)
    vid_callback = VideoCallback(vid_env, eval_freq=6000)
    callbacks = [eval_callback, vid_callback] if vid_env is not None else eval_callback

    # Instantiate Algorithm
    model = PPO("MlpPolicy", train_env, batch_size=(args.num_envs*args.num_steps)//4, use_sde=False, n_steps= args.num_steps, n_epochs=args.num_epochs, gamma=.99, gae_lambda=.95,
                target_kl=.01, verbose=True,  clip_range=.1, ent_coef=.000585045, vf_coef=0.871923, max_grad_norm=10, learning_rate=1e-4,
                tensorboard_log=args.log_dir + 'tf/', policy_kwargs=POLICY_KWARGS)

    # Train
    model.learn(total_timesteps=args.time_steps, callback=callbacks, tb_log_name=args.exp_name)

    run.finish()

